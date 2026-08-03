from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .database import Database, utcnow
from .mcp_client import McpManager
from .providers import ProviderUnavailable
from .research_pipeline import ResearchPipeline
from .settings import Settings


TERMINAL = {"completed", "failed", "cancelled", "partial"}


class ProductRuntime:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.mcp = McpManager()
        self.pipeline = ResearchPipeline(db, settings, self.mcp)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_tasks: list[asyncio.Task] = []
        self.scheduler = AsyncIOScheduler(timezone="UTC")

    async def start(self) -> None:
        self.db.init()
        await self.mcp.load_and_connect()
        self.db.execute("UPDATE tasks SET status='queued', updated_at=? WHERE status='running'", (utcnow(),))
        for row in self.db.fetchall("SELECT id FROM tasks WHERE status='queued' ORDER BY created_at"):
            await self.queue.put(row["id"])
        self.worker_tasks = [asyncio.create_task(self._worker(index), name=f"insightpilot-worker-{index}") for index in range(max(1, self.settings.workers))]
        self.scheduler.add_job(self.dispatch_due_monitors, "interval", seconds=20, id="monitor-dispatch", max_instances=1, coalesce=True)
        self.scheduler.start()

    async def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        for task in self.worker_tasks:
            task.cancel()
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        await self.mcp.disconnect_all()

    async def submit(self, goal: str, project_id: str = "default", mode: str = "research") -> str:
        task_id = self.db.create_task(goal, project_id, mode)
        await self.queue.put(task_id)
        return task_id

    async def _worker(self, index: int) -> None:
        while True:
            task_id = await self.queue.get()
            try:
                task = self.db.task(task_id)
                if not task or task["status"] == "cancelled":
                    continue
                self.db.execute("UPDATE tasks SET attempt_count=attempt_count+1, updated_at=? WHERE id=?", (utcnow(), task_id))
                await self.pipeline.run(task_id)
                await self._finish_monitor_run(task_id, None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                task = self.db.task(task_id)
                if task and task["status"] != "cancelled":
                    self.db.update_task(task_id, "failed", error=str(exc))
                    self.db.add_event(task_id, "task.failed", "supervisor", str(exc), {"error_type": type(exc).__name__, "worker": index})
                await self._finish_monitor_run(task_id, str(exc))
            finally:
                self.queue.task_done()

    async def dispatch_due_monitors(self) -> None:
        now = utcnow()
        monitors = self.db.fetchall("SELECT * FROM monitors WHERE enabled=1 AND next_run_at<=?", (now,))
        for monitor in monitors:
            task_id = await self.submit(monitor["query"], project_id=f"monitor:{monitor['id']}", mode="monitor")
            run_id = str(uuid.uuid4())
            self.db.insert("monitor_runs", {"id": run_id, "monitor_id": monitor["id"], "task_id": task_id, "status": "running", "created_at": utcnow()})
            next_run = datetime.now(timezone.utc) + timedelta(minutes=monitor["interval_minutes"])
            self.db.execute("UPDATE monitors SET last_run_at=?, next_run_at=?, updated_at=? WHERE id=?", (utcnow(), next_run.isoformat(), utcnow(), monitor["id"]))
            self.db.add_event(task_id, "monitor.started", "monitor", f"定时监控已触发：{monitor['name']}", {"monitor_id": monitor["id"], "run_id": run_id})

    async def _finish_monitor_run(self, task_id: str, error: str | None) -> None:
        run = self.db.fetchone("SELECT mr.*, m.last_fingerprint, m.name, m.notify_channel FROM monitor_runs mr JOIN monitors m ON m.id=mr.monitor_id WHERE mr.task_id=?", (task_id,))
        if not run:
            return
        if error:
            self.db.execute("UPDATE monitor_runs SET status='failed', error=?, finished_at=? WHERE id=?", (error, utcnow(), run["id"]))
            return
        sources = self.db.fetchall("SELECT url, content_hash FROM sources WHERE task_id=? ORDER BY url", (task_id,))
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        changed = bool(run["last_fingerprint"] and run["last_fingerprint"] != fingerprint)
        self.db.execute("UPDATE monitor_runs SET status='completed', changed=?, fingerprint=?, finished_at=? WHERE id=?", (int(changed), fingerprint, utcnow(), run["id"]))
        self.db.execute("UPDATE monitors SET last_fingerprint=?, updated_at=? WHERE id=?", (fingerprint, utcnow(), run["monitor_id"]))
        if changed and run.get("notify_channel"):
            approval_id = str(uuid.uuid4())
            payload = {"channel": run["notify_channel"], "monitor_name": run["name"], "task_id": task_id, "text": f"InsightPilot 检测到监控任务“{run['name']}”出现新变化。任务 ID：{task_id}"}
            self.db.insert("approvals", {"id": approval_id, "action_type": "send_notification", "status": "pending", "summary": payload["text"], "payload_json": json.dumps(payload, ensure_ascii=False), "requested_at": utcnow()})
            self.db.add_event(task_id, "approval.requested", "monitor", "检测到变化，外部通知等待人工审批", {"approval_id": approval_id})

    def create_monitor(self, name: str, query: str, interval_minutes: int, notify_channel: str | None) -> str:
        monitor_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self.db.insert("monitors", {
            "id": monitor_id, "name": name, "query": query, "interval_minutes": interval_minutes,
            "enabled": 1, "notify_channel": notify_channel, "last_fingerprint": None, "last_run_at": None,
            "next_run_at": now.isoformat(), "created_at": now.isoformat(), "updated_at": now.isoformat(),
        })
        return monitor_id

    async def decide_approval(self, approval_id: str, approve: bool, note: str | None = None) -> dict[str, Any]:
        approval = self.db.fetchone("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if not approval:
            raise ValueError("approval not found")
        if approval["status"] != "pending":
            raise ValueError("approval already decided")
        status = "approved" if approve else "rejected"
        self.db.execute("UPDATE approvals SET status=?, decided_at=?, decision_note=? WHERE id=?", (status, utcnow(), note, approval_id))
        result: dict[str, Any] = {"status": status}
        if approve and approval["action_type"] == "send_notification":
            payload = json.loads(approval["payload_json"])
            result["delivery"] = await self._send_notification(payload)
        return result

    async def _send_notification(self, payload: dict[str, Any]) -> dict[str, Any]:
        channel = payload.get("channel")
        if channel != "feishu":
            raise ProviderUnavailable(f"暂不支持通知渠道：{channel}")
        if not self.settings.feishu_webhook_url:
            raise ProviderUnavailable("FEISHU_WEBHOOK_URL 未配置")
        body = {"msg_type": "text", "content": {"text": payload["text"]}}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(self.settings.feishu_webhook_url, json=body)
            response.raise_for_status()
            return {"channel": channel, "http_status": response.status_code}
