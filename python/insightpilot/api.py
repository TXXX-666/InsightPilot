from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .api_auth import require_api_access
from .database import Database
from .research_pipeline import evidence_graph
from .runtime import ProductRuntime, TERMINAL
from .settings import settings


settings.ensure_dirs()
db = Database(settings.db_path)
runtime = ProductRuntime(db, settings)


def browser_status() -> dict[str, str]:
    if not importlib.util.find_spec("playwright"):
        return {"http_fetch": "ready", "playwright": "package_missing"}
    cache = Path(os.getenv("PLAYWRIGHT_BROWSERS_PATH") or (Path(os.getenv("LOCALAPPDATA", "")) / "ms-playwright"))
    executables = list(cache.glob("chromium-*/*/chrome.exe")) + list(cache.glob("chromium_headless_shell-*/*/headless_shell.exe"))
    return {"http_fetch": "ready", "playwright": "runtime_installed_unverified" if executables else "browser_runtime_missing"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(
    title="InsightPilot API",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_access)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskCreate(BaseModel):
    goal: str = Field(min_length=5, max_length=4000)
    project_id: str = Field(default="default", max_length=100)


class MonitorCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    query: str = Field(min_length=5, max_length=4000)
    interval_minutes: int = Field(default=1440, ge=1, le=525600)
    notify_channel: str | None = Field(default=None, pattern="^(feishu)?$")


class ApprovalDecision(BaseModel):
    approve: bool
    note: str | None = Field(default=None, max_length=1000)


@app.get("/api/v1/health")
async def health():
    return {
        "status": "running",
        "database": {"status": "ready"},
        "queue": {"pending": runtime.queue.qsize(), "workers": len(runtime.worker_tasks)},
        "providers": {
            "search": await runtime.pipeline.search.health(),
            "llm": await runtime.pipeline.llm.health(),
            "browser": browser_status(),
            "notification": {"provider": "feishu", "configured": bool(settings.feishu_webhook_url)},
            "mcp": runtime.mcp.status(),
        },
        "truthfulness": "未配置的外部服务会明确报错，不会返回伪实时结果",
        "architecture": "supervisor-blackboard-multi-agent",
    }


@app.get("/api/v1/mcp/status")
async def mcp_status():
    return runtime.mcp.status()


@app.get("/api/v1/mcp/tools")
async def mcp_tools():
    return runtime.mcp.get_tool_definitions()


@app.get("/api/v1/metrics")
async def metrics():
    task_counts = db.fetchall("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status")
    totals = db.fetchone("SELECT (SELECT COUNT(*) FROM sources) AS sources, (SELECT COUNT(*) FROM evidence) AS evidence, (SELECT COUNT(*) FROM claims) AS claims, (SELECT COUNT(*) FROM agent_runs) AS agent_runs, (SELECT COUNT(*) FROM workflow_checkpoints) AS checkpoints, (SELECT COUNT(*) FROM monitors WHERE enabled=1) AS active_monitors, (SELECT COUNT(*) FROM approvals WHERE status='pending') AS pending_approvals")
    return {"tasks": {row["status"]: row["count"] for row in task_counts}, "totals": totals, "queue_depth": runtime.queue.qsize(), "workers": len(runtime.worker_tasks)}


@app.post("/api/v1/tasks", status_code=202)
async def create_task(body: TaskCreate):
    task_id = await runtime.submit(body.goal.strip(), body.project_id)
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/v1/tasks")
async def list_tasks(limit: int = Query(default=50, ge=1, le=200)):
    return db.list_tasks(limit)


@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str):
    task = db.task(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    return task


@app.post("/api/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    task = db.task(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    if task["status"] in TERMINAL:
        return {"task_id": task_id, "status": task["status"]}
    db.update_task(task_id, "cancelled")
    db.add_event(task_id, "task.cancelled", "user", "用户取消了任务")
    return {"task_id": task_id, "status": "cancelled"}


@app.post("/api/v1/tasks/{task_id}/retry", status_code=202)
async def retry_task(task_id: str):
    task = db.task(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    if task["status"] not in {"failed", "partial", "cancelled"}:
        raise HTTPException(409, "only failed, partial, or cancelled tasks can be retried")
    db.execute("UPDATE tasks SET status='queued', error=NULL, finished_at=NULL, updated_at=? WHERE id=?", (__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), task_id))
    db.add_event(task_id, "task.retry_queued", "user", "任务已重新入队，将从已持久化阶段恢复")
    await runtime.queue.put(task_id)
    return {"task_id": task_id, "status": "queued", "resume": True}


@app.get("/api/v1/tasks/{task_id}/events")
async def task_events(task_id: str, after_id: int = Query(default=0, ge=0)):
    if not db.task(task_id):
        raise HTTPException(404, "task not found")
    return db.fetchall("SELECT * FROM events WHERE task_id=? AND id>? ORDER BY id", (task_id, after_id))


@app.get("/api/v1/tasks/{task_id}/agent-trace")
async def task_agent_trace(task_id: str):
    if not db.task(task_id):
        raise HTTPException(404, "task not found")
    return db.agent_trace(task_id)


@app.get("/api/v1/tasks/{task_id}/events/stream")
async def task_event_stream(task_id: str, after_id: int = Query(default=0, ge=0)):
    if not db.task(task_id):
        raise HTTPException(404, "task not found")

    async def generate() -> AsyncIterator[str]:
        cursor = after_id
        idle_terminal_ticks = 0
        while True:
            rows = db.fetchall("SELECT * FROM events WHERE task_id=? AND id>? ORDER BY id", (task_id, cursor))
            for row in rows:
                cursor = row["id"]
                yield f"id: {cursor}\nevent: {row['event_type']}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
            task = db.task(task_id)
            if task and task["status"] in TERMINAL and not rows:
                idle_terminal_ticks += 1
                if idle_terminal_ticks >= 2:
                    break
            else:
                idle_terminal_ticks = 0
            yield ": heartbeat\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/v1/tasks/{task_id}/evidence-graph")
async def task_evidence_graph(task_id: str):
    if not db.task(task_id):
        raise HTTPException(404, "task not found")
    return evidence_graph(db, task_id)


@app.get("/api/v1/tasks/{task_id}/report")
async def task_report(task_id: str, format: str = Query(default="md", pattern="^(md|docx|pdf)$")):
    task = db.task(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    base_path = task.get("report_path")
    path = str(Path(base_path).with_suffix(f".{format}")) if base_path else None
    if not path or not Path(path).is_file():
        raise HTTPException(404, "report not ready")
    media = {"md": "text/markdown", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "pdf": "application/pdf"}[format]
    return FileResponse(path, media_type=media, filename=f"insightpilot-{task_id}.{format}")


@app.post("/api/v1/monitors", status_code=201)
async def create_monitor(body: MonitorCreate):
    monitor_id = runtime.create_monitor(body.name.strip(), body.query.strip(), body.interval_minutes, body.notify_channel or None)
    return {"monitor_id": monitor_id, "status": "scheduled"}


@app.get("/api/v1/monitors")
async def list_monitors():
    return db.fetchall("SELECT * FROM monitors ORDER BY created_at DESC")


@app.patch("/api/v1/monitors/{monitor_id}/toggle")
async def toggle_monitor(monitor_id: str):
    monitor = db.fetchone("SELECT * FROM monitors WHERE id=?", (monitor_id,))
    if not monitor:
        raise HTTPException(404, "monitor not found")
    enabled = 0 if monitor["enabled"] else 1
    db.execute("UPDATE monitors SET enabled=?, updated_at=? WHERE id=?", (enabled, __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), monitor_id))
    return {"monitor_id": monitor_id, "enabled": bool(enabled)}


@app.get("/api/v1/approvals")
async def list_approvals(status: str = Query(default="pending")):
    return db.fetchall("SELECT * FROM approvals WHERE status=? ORDER BY requested_at DESC", (status,))


@app.post("/api/v1/approvals/{approval_id}/decision")
async def decide_approval(approval_id: str, body: ApprovalDecision):
    try:
        return await runtime.decide_approval(approval_id, body.approve, body.note)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


def main() -> None:
    uvicorn.run("insightpilot.api:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
