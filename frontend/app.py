from __future__ import annotations

import hmac
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
API = os.getenv("INSIGHTPILOT_API_URL", "http://127.0.0.1:8000")
EXPECTED_API_TOKEN = os.getenv("INSIGHTPILOT_API_TOKEN", "").strip()


def display_time(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def api(method: str, path: str, **kwargs):
    try:
        headers = dict(kwargs.pop("headers", {}))
        api_token = st.session_state.get("insightpilot_api_token", "")
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        response = httpx.request(
            method,
            API + path,
            timeout=30,
            headers=headers,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API 请求失败：{exc}")
        return None


st.set_page_config(page_title="InsightPilot", page_icon="🔎", layout="wide")

if EXPECTED_API_TOKEN and not st.session_state.get("insightpilot_authenticated"):
    st.title("InsightPilot")
    st.caption("请输入访问令牌后进入研究平台")
    with st.form("insightpilot-login"):
        supplied_token = st.text_input("访问令牌", type="password")
        if st.form_submit_button("登录", type="primary"):
            normalized_token = supplied_token.strip()
            if normalized_token and hmac.compare_digest(
                normalized_token, EXPECTED_API_TOKEN
            ):
                st.session_state["insightpilot_authenticated"] = True
                st.session_state["insightpilot_api_token"] = normalized_token
                st.rerun()
            else:
                st.error("访问令牌不正确")
    st.stop()

if EXPECTED_API_TOKEN:
    with st.sidebar:
        if st.button("退出登录"):
            st.session_state.pop("insightpilot_authenticated", None)
            st.session_state.pop("insightpilot_api_token", None)
            st.rerun()

st.title("InsightPilot · 实时智能情报研究平台")
st.caption("真实联网检索 · 多智能体核验 · 证据图 · 定时监控 · 人工审批")

health = api("GET", "/api/v1/health")
if health:
    providers = health["providers"]
    cols = st.columns(5)
    cols[0].metric("后端", health["status"])
    cols[1].metric("实时搜索", providers["search"]["status"])
    cols[2].metric("大模型", providers["llm"]["status"])
    cols[3].metric("队列任务", health["queue"]["pending"])
    mcp = providers.get("mcp", {})
    cols[4].metric("MCP", f"{mcp.get('connected', 0)}/{mcp.get('configured', 0)}")
    if not providers["search"]["configured"] or not providers["llm"]["configured"]:
        st.warning("当前外部服务未完整配置。系统会拒绝伪实时结果；请按 .env.example 配置 Tavily 和模型 API。")
    if providers["browser"]["playwright"] in {"package_missing", "browser_runtime_missing"}:
        st.info("JavaScript 动态网页浏览器未完整安装；普通网页实时抓取不受影响。可运行 scripts/install-browser.ps1。")
    for server in mcp.get("servers", []):
        if server.get("status") == "error":
            st.warning(f"MCP {server['name']} 连接失败：{server.get('error')}")

research_tab, monitor_tab, approval_tab = st.tabs(["实时研究", "长期监控", "审批中心"])

with research_tab:
    with st.form("new-research"):
        goal = st.text_area("研究目标", placeholder="例如：调研近 30 天国内大模型应用实习岗位的技能要求，并给出带来源的趋势报告", height=110)
        project_id = st.text_input("项目空间", value="default")
        submitted = st.form_submit_button("启动后台研究", type="primary")
        if submitted and goal.strip():
            created = api("POST", "/api/v1/tasks", json={"goal": goal, "project_id": project_id})
            if created:
                st.session_state["task_id"] = created["task_id"]
                st.success(f"任务已创建：{created['task_id']}")

    tasks = api("GET", "/api/v1/tasks?limit=30") or []
    if tasks:
        options = {f"{display_time(t['created_at'])} · {t['status']} · {t['goal'][:60]}": t["id"] for t in tasks}
        labels = list(options)
        current_id = st.session_state.get("task_id")
        default_index = next((i for i, label in enumerate(labels) if options[label] == current_id), 0)
        selected = st.selectbox("查看任务", labels, index=default_index)
        st.session_state["task_id"] = options[selected]

        @st.fragment(run_every=2)
        def live_task(task_id: str):
            task = api("GET", f"/api/v1/tasks/{task_id}")
            events = api("GET", f"/api/v1/tasks/{task_id}/events") or []
            trace = api("GET", f"/api/v1/tasks/{task_id}/agent-trace") or {}
            if not task:
                return
            c1, c2, c3 = st.columns(3)
            c1.metric("状态", task["status"])
            c2.metric("事件数", len(events))
            c3.metric("更新时间", display_time(task["updated_at"]))
            if task.get("error"):
                st.error(task["error"])
            failed_runs = [run for run in trace.get("runs", []) if run.get("status") == "failed"]
            if task["status"] == "partial" and failed_runs:
                latest_failure = failed_runs[-1]
                st.error(f"最近失败节点：{latest_failure['agent']} · {latest_failure.get('error') or '未记录错误'}")
            if task["status"] in {"failed", "partial", "cancelled"}:
                if st.button("从失败节点恢复", key=f"retry-{task_id}", type="primary"):
                    retried = api("POST", f"/api/v1/tasks/{task_id}/retry")
                    if retried:
                        st.rerun()
            if task["status"] not in {"completed", "failed", "cancelled", "partial"}:
                if st.button("取消任务", key=f"cancel-{task_id}"):
                    api("POST", f"/api/v1/tasks/{task_id}/cancel")
            if events:
                st.subheader("实时 Agent 事件")
                frame = pd.DataFrame([{"时间": display_time(e["created_at"]), "Agent": e["agent"], "事件": e["event_type"], "说明": e["message"]} for e in events[::-1]])
                st.dataframe(frame, width="stretch", hide_index=True)
            if trace and trace.get("runs"):
                st.subheader("Multi-Agent 执行轨迹")
                trace_rows = []
                for run in trace["runs"]:
                    output = run.get("output") or {}
                    trace_rows.append({
                        "Agent": run["agent"],
                        "状态": run["status"],
                        "轮次": run["round"],
                        "允许工具": "、".join(run.get("allowed_tools", [])) or "无",
                        "建议动作": output.get("proposed_action") or output.get("next_agent", ""),
                        "原因": output.get("reason", ""),
                    })
                st.dataframe(pd.DataFrame(trace_rows), width="stretch", hide_index=True)
                checkpoint = trace.get("checkpoint") or {}
                state = checkpoint.get("state") or {}
                if state:
                    q1, q2, q3, q4 = st.columns(4)
                    q1.metric("检索轮次", state.get("search_round", 0))
                    q2.metric("证据覆盖", f"{float(state.get('coverage_score', 0)):.2f}")
                    q3.metric("引用覆盖", f"{float(state.get('citation_score', 0)):.2f}")
                    q4.metric("下一 Agent", checkpoint.get("current_agent", "-"))
            if task["status"] in {"completed", "partial"}:
                result = task.get("result") or {}
                st.subheader("研究报告")
                if task["status"] == "partial":
                    reason = result.get("partial_reason") or "任务达到质量、循环或 Provider 边界"
                    st.warning(f"部分报告原因：{reason}")
                elif result.get("completion_type") == "completed_with_limitations":
                    st.info("研究质量门槛已通过，报告已完成；以下局限不阻断核心结论。")
                limitations = result.get("limitations") or []
                if limitations:
                    with st.expander("研究局限", expanded=task["status"] == "partial"):
                        for item in limitations:
                            st.markdown(f"- {item}")
                st.markdown(result.get("report", ""))
                st.download_button("下载 Markdown 报告", result.get("report", ""), file_name=f"insightpilot-{task_id}.md")
                st.caption("Word/PDF 报告可通过 API 下载：/api/v1/tasks/{task_id}/report?format=docx|pdf")
                graph = api("GET", f"/api/v1/tasks/{task_id}/evidence-graph")
                if graph:
                    st.subheader("证据追溯")
                    verified = sum(1 for claim in graph["claims"] if claim.get("status") == "verified")
                    mixed = sum(1 for claim in graph["claims"] if claim.get("status") == "mixed")
                    st.write(
                        f"主张 {len(graph['claims'])} 个（verified {verified}，mixed {mixed}），"
                        f"证据 {len(graph['evidence'])} 条，关联 {len(graph['links'])} 条"
                    )
                    st.dataframe(pd.DataFrame(graph["claims"]), width="stretch", hide_index=True)

        live_task(st.session_state["task_id"])

with monitor_tab:
    with st.form("new-monitor"):
        name = st.text_input("监控名称", placeholder="LLM 实习岗位监控")
        query = st.text_area("监控目标", placeholder="每天追踪新增的大模型应用开发实习岗位，并分析技能要求变化")
        interval = st.number_input("执行间隔（分钟）", min_value=1, value=1440)
        notify = st.selectbox("变化通知", ["不通知", "飞书（需审批）"])
        if st.form_submit_button("创建监控"):
            channel = "feishu" if notify.startswith("飞书") else None
            created = api("POST", "/api/v1/monitors", json={"name": name, "query": query, "interval_minutes": interval, "notify_channel": channel})
            if created:
                st.success(f"监控已创建：{created['monitor_id']}")
    monitors = api("GET", "/api/v1/monitors") or []
    if monitors:
        st.dataframe(pd.DataFrame(monitors), width="stretch", hide_index=True)

with approval_tab:
    approvals = api("GET", "/api/v1/approvals?status=pending") or []
    if not approvals:
        st.info("当前没有等待审批的外部操作。")
    for item in approvals:
        with st.container(border=True):
            st.write(item["summary"])
            st.caption(f"类型：{item['action_type']} · 请求时间：{display_time(item['requested_at'])}")
            left, right = st.columns(2)
            if left.button("批准并执行", key=f"approve-{item['id']}", type="primary"):
                api("POST", f"/api/v1/approvals/{item['id']}/decision", json={"approve": True})
                st.rerun()
            if right.button("拒绝", key=f"reject-{item['id']}"):
                api("POST", f"/api/v1/approvals/{item['id']}/decision", json={"approve": False})
                st.rerun()
