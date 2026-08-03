from __future__ import annotations

import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st


API = os.getenv("INSIGHTPILOT_API_URL", "http://127.0.0.1:8000")


def api(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, API + path, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API 请求失败：{exc}")
        return None


st.set_page_config(page_title="InsightPilot", page_icon="🔎", layout="wide")
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
        options = {f"{t['created_at'][:19]} · {t['status']} · {t['goal'][:60]}": t["id"] for t in tasks}
        labels = list(options)
        current_id = st.session_state.get("task_id")
        default_index = next((i for i, label in enumerate(labels) if options[label] == current_id), 0)
        selected = st.selectbox("查看任务", labels, index=default_index)
        st.session_state["task_id"] = options[selected]

        @st.fragment(run_every=2)
        def live_task(task_id: str):
            task = api("GET", f"/api/v1/tasks/{task_id}")
            events = api("GET", f"/api/v1/tasks/{task_id}/events") or []
            if not task:
                return
            c1, c2, c3 = st.columns(3)
            c1.metric("状态", task["status"])
            c2.metric("事件数", len(events))
            c3.metric("更新时间", task["updated_at"][:19])
            if task.get("error"):
                st.error(task["error"])
                if task["status"] == "failed" and st.button("从断点重试", key=f"retry-{task_id}", type="primary"):
                    retried = api("POST", f"/api/v1/tasks/{task_id}/retry")
                    if retried:
                        st.rerun()
            if task["status"] not in {"completed", "failed", "cancelled", "partial"}:
                if st.button("取消任务", key=f"cancel-{task_id}"):
                    api("POST", f"/api/v1/tasks/{task_id}/cancel")
            if events:
                st.subheader("实时 Agent 事件")
                frame = pd.DataFrame([{"时间": e["created_at"][:19], "Agent": e["agent"], "事件": e["event_type"], "说明": e["message"]} for e in events[::-1]])
                st.dataframe(frame, use_container_width=True, hide_index=True)
            if task["status"] == "completed":
                result = task.get("result") or {}
                st.subheader("研究报告")
                st.markdown(result.get("report", ""))
                st.download_button("下载 Markdown 报告", result.get("report", ""), file_name=f"insightpilot-{task_id}.md")
                st.caption("Word/PDF 报告可通过 API 下载：/api/v1/tasks/{task_id}/report?format=docx|pdf")
                graph = api("GET", f"/api/v1/tasks/{task_id}/evidence-graph")
                if graph:
                    st.subheader("证据追溯")
                    st.write(f"主张 {len(graph['claims'])} 个，证据 {len(graph['evidence'])} 条，关联 {len(graph['links'])} 条")
                    st.dataframe(pd.DataFrame(graph["claims"]), use_container_width=True, hide_index=True)

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
        st.dataframe(pd.DataFrame(monitors), use_container_width=True, hide_index=True)

with approval_tab:
    approvals = api("GET", "/api/v1/approvals?status=pending") or []
    if not approvals:
        st.info("当前没有等待审批的外部操作。")
    for item in approvals:
        with st.container(border=True):
            st.write(item["summary"])
            st.caption(f"类型：{item['action_type']} · 请求时间：{item['requested_at']}")
            left, right = st.columns(2)
            if left.button("批准并执行", key=f"approve-{item['id']}", type="primary"):
                api("POST", f"/api/v1/approvals/{item['id']}/decision", json={"approve": True})
                st.rerun()
            if right.button("拒绝", key=f"reject-{item['id']}"):
                api("POST", f"/api/v1/approvals/{item['id']}/decision", json={"approve": False})
                st.rerun()
