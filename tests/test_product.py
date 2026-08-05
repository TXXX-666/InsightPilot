from __future__ import annotations

import json
import importlib.util
from dataclasses import replace

import pytest
from fastapi import HTTPException

from insightpilot.database import Database, utcnow
from insightpilot.academic_research import ArxivMcpSearch, is_academic_goal, normalize_arxiv_response
from insightpilot.providers import LLMGateway, ProviderUnavailable
from insightpilot.research_pipeline import (
    ResearchPipeline,
    evidence_graph,
    reliability_for,
    sha256,
    source_type,
)
from insightpilot.report_export import markdown_to_docx, markdown_to_pdf
from insightpilot.runtime import ProductRuntime
from insightpilot.network_security import UnsafeURL, assert_public_url
from insightpilot.mcp_client import McpConnection, McpDependencyMissing, McpManager, _safe_error
from insightpilot.api_auth import validate_api_access
from insightpilot.settings import settings


@pytest.fixture()
def database(tmp_path):
    db = Database(tmp_path / "test.db")
    db.init()
    return db


def test_task_and_events_are_persistent(database):
    task_id = database.create_task("research a current market change", "test")
    database.update_task(task_id, "running")
    database.add_event(task_id, "agent.started", "search-planner", "planning")
    assert database.task(task_id)["status"] == "running"
    events = database.fetchall("SELECT * FROM events WHERE task_id=? ORDER BY id", (task_id,))
    assert [event["event_type"] for event in events] == ["task.queued", "agent.started"]


def test_api_token_authentication():
    validate_api_access("secret-token", "Bearer secret-token", "203.0.113.10")

    with pytest.raises(HTTPException) as missing:
        validate_api_access("secret-token", None, "127.0.0.1")
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as incorrect:
        validate_api_access("secret-token", "Bearer wrong-token", "127.0.0.1")
    assert incorrect.value.status_code == 401


def test_api_without_token_is_loopback_only():
    validate_api_access("", None, "127.0.0.1")
    validate_api_access("", None, "::1")

    with pytest.raises(HTTPException) as remote:
        validate_api_access("", None, "203.0.113.10")
    assert remote.value.status_code == 403


@pytest.mark.asyncio
async def test_missing_search_key_fails_explicitly(database, tmp_path):
    config = replace(settings, db_path=tmp_path / "test.db", report_dir=tmp_path / "reports", tavily_api_key="", llm_api_key="test")
    task_id = database.create_task("research current AI application jobs")
    pipeline = ResearchPipeline(database, config)
    with pytest.raises(ProviderUnavailable, match="TAVILY_API_KEY"):
        await pipeline.run(task_id)
    assert database.fetchall("SELECT * FROM sources WHERE task_id=?", (task_id,)) == []


@pytest.mark.asyncio
async def test_llm_json_retries_after_empty_response():
    config = replace(settings, llm_api_key="test", llm_model="test", json_retries=1)
    gateway = LLMGateway(config)
    responses = iter(["", '{"claims": []}'])

    async def fake_complete(*args, **kwargs):
        return next(responses)

    gateway.complete = fake_complete
    assert await gateway.json("system", "user") == {"claims": []}


@pytest.mark.asyncio
async def test_llm_json_reports_actionable_error_after_exhausted_retries():
    config = replace(settings, llm_api_key="test", llm_model="test", json_retries=1)
    gateway = LLMGateway(config)

    async def fake_complete(*args, **kwargs):
        return ""

    gateway.complete = fake_complete
    with pytest.raises(ProviderUnavailable, match="未返回有效 JSON"):
        await gateway.json("system", "user")


@pytest.mark.asyncio
async def test_llm_json_relaxes_mode_and_increases_budget_after_empty_response():
    config = replace(
        settings,
        llm_api_key="test",
        llm_model="test",
        json_retries=1,
        json_max_tokens=1000,
        json_retry_max_tokens=4000,
    )
    gateway = LLMGateway(config)
    calls = []

    async def fake_complete(*args, **kwargs):
        calls.append(kwargs)
        return "" if len(calls) == 1 else '{"ok": true}'

    gateway.complete = fake_complete
    assert await gateway.json("system", "user") == {"ok": True}
    assert calls == [
        {"json_mode": True, "max_tokens": 1000},
        {"json_mode": False, "max_tokens": 2000},
    ]


@pytest.mark.asyncio
async def test_report_writer_falls_back_when_stream_is_empty(database, tmp_path):
    config = replace(settings, db_path=tmp_path / "test.db", report_dir=tmp_path / "reports")
    pipeline = ResearchPipeline(database, config)
    calls = []

    async def fake_complete(*args, **kwargs):
        calls.append(kwargs)
        return "" if kwargs.get("stream") else "# 研究报告\n\n内容"

    pipeline.llm.complete = fake_complete
    report = await pipeline._write_report(
        database.create_task("test report"),
        "test report",
        [{"evidence_ids": ["evidence-1"]}],
        [{"evidence_id": "evidence-1", "title": "Source", "url": "https://example.com", "published_at": None, "retrieved_at": "now", "quote": "quote", "reliability": 0.8}],
        {"gaps": [], "risks": [], "overall_confidence": 0.8},
    )
    assert report == "# 研究报告\n\n内容"
    assert [call.get("stream", False) for call in calls] == [True, False]
    assert all(call["max_tokens"] == config.report_max_tokens for call in calls)


@pytest.mark.asyncio
async def test_report_writer_creates_evidence_fallback_after_two_empty_responses(database, tmp_path):
    config = replace(settings, db_path=tmp_path / "test.db", report_dir=tmp_path / "reports")
    pipeline = ResearchPipeline(database, config)

    async def fake_complete(*args, **kwargs):
        return ""

    pipeline.llm.complete = fake_complete
    task_id = database.create_task("test report")
    report = await pipeline._write_report(
        task_id,
        "test report",
        [{"text": "A verified finding", "evidence_ids": ["evidence-1"]}],
        [{"evidence_id": "evidence-1", "title": "Source", "url": "https://example.com", "published_at": None, "retrieved_at": "now", "quote": "quote", "reliability": 0.8}],
        {"gaps": ["Need more sources"], "risks": [], "overall_confidence": 0.8},
    )
    assert "A verified finding [evidence-1]" in report
    assert "https://example.com" in report
    event = database.fetchone("SELECT payload_json FROM events WHERE task_id=? AND agent='report-writer' AND event_type='agent.completed'", (task_id,))
    assert json.loads(event["payload_json"])["generation"] == "evidence_fallback"


@pytest.mark.asyncio
async def test_report_without_verified_claims_lists_retrieved_evidence_accurately(database, tmp_path):
    config = replace(settings, db_path=tmp_path / "test.db", report_dir=tmp_path / "reports")
    pipeline = ResearchPipeline(database, config)

    async def unexpected_completion(*args, **kwargs):
        raise AssertionError("LLM should not write conclusions without verified claims")

    pipeline.llm.complete = unexpected_completion
    task_id = database.create_task("inspect a market")
    report = await pipeline._write_report(
        task_id,
        "inspect a market",
        [],
        [{
            "evidence_id": "evidence-1",
            "source_id": "source-1",
            "title": "Candidate source",
            "url": "https://example.com/source",
            "retrieved_at": "2026-08-05T00:00:00+00:00",
            "fetch_status": "fetched",
        }],
        {"gaps": ["verification failed"], "risks": []},
        partial=True,
    )
    assert "已提取 1 条候选证据" in report
    assert "https://example.com/source" in report
    assert "不是检索结果为零" in report


def test_source_quality_distinguishes_primary_and_weak_sources():
    official = "https://example.gov.cn/jobs/llm"
    community = "https://blog.csdn.net/example/article/details/1"
    assert source_type(official) == "official"
    assert source_type(community) == "community"
    assert reliability_for(official, 0.8, "fetched") > reliability_for(
        community, 0.8, "fetched"
    )
    assert reliability_for(official, 0.8, "fetched") > reliability_for(
        official, 0.8, "search_snippet"
    )


def test_source_quality_gate_downgrades_community_only_claim(database, tmp_path):
    pipeline = ResearchPipeline(
        database,
        replace(settings, db_path=tmp_path / "test.db", report_dir=tmp_path / "reports"),
    )
    task_id = database.create_task("quality gate")
    database.insert("sources", {
        "id": "s1", "task_id": task_id,
        "url": "https://blog.csdn.net/example/article/details/1", "title": "Community",
        "published_at": None, "retrieved_at": utcnow(), "source_type": "community",
        "fetch_status": "fetched", "content_hash": sha256("source"), "raw_text": "text",
    })
    database.insert("evidence", {
        "id": "e1", "task_id": task_id, "source_id": "s1", "claim": None,
        "quote": "A sufficiently long community quotation for the quality gate.",
        "reliability": 0.8, "stance": "supports", "content_hash": sha256("evidence"),
        "created_at": utcnow(),
    })
    database.insert("claims", {
        "id": "c1", "task_id": task_id, "text": "Community-only claim",
        "confidence": 0.9, "status": "verified", "created_at": utcnow(),
    })
    database.insert("claim_evidence", {
        "claim_id": "c1", "evidence_id": "e1", "relation": "supports",
    })
    claims = pipeline._apply_source_quality_gate(task_id, pipeline._load_claims(task_id))
    assert claims[0]["status"] == "mixed"
    assert database.fetchone("SELECT status FROM claims WHERE id='c1'")["status"] == "mixed"


def test_evidence_graph_keeps_real_links(database):
    task_id = database.create_task("verify a claim")
    source_id = "source-1"
    evidence_id = "evidence-1"
    claim_id = "claim-1"
    database.insert("sources", {"id": source_id, "task_id": task_id, "url": "https://example.com/source", "title": "Source", "published_at": None, "retrieved_at": utcnow(), "source_type": "web", "content_hash": sha256("source"), "raw_text": "source"})
    database.insert("evidence", {"id": evidence_id, "task_id": task_id, "source_id": source_id, "claim": None, "quote": "An attributable quotation long enough to be useful.", "reliability": 0.7, "stance": "supports", "content_hash": sha256("quote"), "created_at": utcnow()})
    database.insert("claims", {"id": claim_id, "task_id": task_id, "text": "A verifiable claim", "confidence": 0.7, "status": "verified", "created_at": utcnow()})
    database.insert("claim_evidence", {"claim_id": claim_id, "evidence_id": evidence_id, "relation": "supports"})
    graph = evidence_graph(database, task_id)
    assert graph["links"] == [{"claim_id": claim_id, "evidence_id": evidence_id, "relation": "supports"}]
    assert graph["evidence"][0]["url"] == "https://example.com/source"


def test_monitor_fingerprint_input_is_deterministic():
    first = json.dumps([{"url": "a", "content_hash": "1"}], sort_keys=True)
    second = json.dumps([{"content_hash": "1", "url": "a"}], sort_keys=True)
    assert sha256(first) == sha256(second)


def test_word_and_pdf_report_export(tmp_path):
    markdown = "# 实时研究报告\n\n## 结论\n这是带有证据引用的测试结论 [evidence-1]。"
    docx = markdown_to_docx(markdown, tmp_path / "report.docx")
    pdf = markdown_to_pdf(markdown, tmp_path / "report.pdf")
    assert docx.stat().st_size > 1000
    assert pdf.stat().st_size > 1000


@pytest.mark.asyncio
async def test_notification_is_not_sent_before_approval(database, tmp_path):
    config = replace(settings, db_path=tmp_path / "test.db", report_dir=tmp_path / "reports", feishu_webhook_url="")
    runtime = ProductRuntime(database, config)
    database.insert("approvals", {"id": "approval-1", "action_type": "send_notification", "status": "pending", "summary": "external message", "payload_json": json.dumps({"channel": "feishu", "text": "external message"}), "requested_at": utcnow()})
    assert database.fetchone("SELECT status FROM approvals WHERE id='approval-1'")["status"] == "pending"
    decision = await runtime.decide_approval("approval-1", approve=False)
    assert decision["status"] == "rejected"
    assert database.fetchone("SELECT status FROM approvals WHERE id='approval-1'")["status"] == "rejected"


@pytest.mark.asyncio
async def test_ssrf_guard_rejects_local_targets():
    with pytest.raises(UnsafeURL):
        await assert_public_url("http://127.0.0.1:8000/private")
    with pytest.raises(UnsafeURL):
        await assert_public_url("http://localhost/admin")


def test_streamable_http_mcp_config_expands_env_and_masks_secret(monkeypatch, tmp_path):
    secret_url = "https://mcp.example.test/private/session-token"
    monkeypatch.setenv("MODELSCOPE_TAVILY_MCP_URL", secret_url)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"modelscope-tavily": {"type": "streamable-http", "url": "${MODELSCOPE_TAVILY_MCP_URL}"}}}), encoding="utf-8")
    manager = McpManager()
    configs = manager._load_configs()
    assert configs["modelscope-tavily"]["url"] == secret_url
    connection = McpConnection("modelscope-tavily", configs["modelscope-tavily"])
    assert connection.transport == "streamable-http"
    assert "session-token" not in connection.endpoint
    assert connection.endpoint == "https://mcp.example.test/***"
    safe_error = _safe_error(RuntimeError(f"failed at {secret_url}"), configs["modelscope-tavily"])
    assert "session-token" not in safe_error


def test_remote_arxiv_mcp_config_can_be_supplied_only_through_env(monkeypatch, tmp_path):
    secret_url = "https://mcp.example.test/arxiv/private-session"
    monkeypatch.setenv("MODELSCOPE_ARXIV_MCP_URL", secret_url)
    monkeypatch.delenv("MODELSCOPE_TAVILY_MCP_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    manager = McpManager()
    configs = manager._load_configs()
    assert configs["modelscope-arxiv"] == {"type": "streamable-http", "url": secret_url}
    status_endpoint = McpConnection("modelscope-arxiv", configs["modelscope-arxiv"]).endpoint
    assert status_endpoint == "https://mcp.example.test/***"
    assert "private-session" not in status_endpoint


def test_academic_goal_detection_and_arxiv_result_normalization():
    assert is_academic_goal("调研最近两年的 RAG 论文和评测基准")
    assert not is_academic_goal("监控某公司的最新产品发布")
    payload = json.dumps({
        "papers": [{
            "arxiv_id": "2401.01234v2",
            "title": "A Reliable RAG Benchmark",
            "authors": ["Alice", "Bob"],
            "abstract": "We introduce a benchmark for retrieval-augmented generation.",
            "published": "2024-01-03",
        }]
    })
    papers = normalize_arxiv_response(payload, "RAG benchmark")
    assert papers == [{
        "url": "https://arxiv.org/abs/2401.01234v2",
        "title": "A Reliable RAG Benchmark",
        "snippet": "Alice, Bob\n\nWe introduce a benchmark for retrieval-augmented generation.",
        "published_at": "2024-01-03",
        "score": 0.96,
        "provider": "mcp:arxiv",
    }]


@pytest.mark.asyncio
async def test_arxiv_adapter_discovers_search_tool_and_uses_its_schema():
    manager = McpManager()
    manager._tools = [{
        "serverName": "modelscope-arxiv",
        "name": "search_papers",
        "description": "Search arXiv papers",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        },
    }]
    captured = {}

    async def fake_call(name, args):
        captured.update(name=name, args=args)
        return '{"results":[{"id":"2502.12345","title":"Agent Research Systems","summary":"Evidence-first research."}]}'

    manager.call_tool = fake_call
    adapter = ArxivMcpSearch(manager)
    papers = await adapter.search("research agents", max_results=5)
    assert captured == {
        "name": "mcp__modelscope-arxiv__search_papers",
        "args": {"query": "research agents", "max_results": 5},
    }
    assert papers[0]["url"] == "https://arxiv.org/abs/2502.12345"


@pytest.mark.asyncio
async def test_academic_pipeline_can_search_arxiv_without_tavily(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "test.db",
        report_dir=tmp_path / "reports",
        tavily_api_key="",
    )
    manager = McpManager()
    manager._tools = [{
        "serverName": "modelscope-arxiv",
        "name": "search_papers",
        "description": "Search arXiv papers",
        "inputSchema": {"properties": {"query": {"type": "string"}}},
    }]

    async def fake_call(name, args):
        return '{"papers":[{"arxiv_id":"2601.00001","title":"Evidence-first Agents","abstract":"A research agent evaluation."}]}'

    manager.call_tool = fake_call
    task_id = database.create_task("调研 research agent 论文")
    pipeline = ResearchPipeline(database, config, manager)
    results = await pipeline._search_all(task_id, ["research agent evaluation"], academic=True)
    assert results[0]["provider"] == "mcp:arxiv"
    event = database.fetchone("SELECT payload_json FROM events WHERE task_id=? AND event_type='search.completed'", (task_id,))
    assert json.loads(event["payload_json"])["transport"] == "streamable-http"


def test_mcp_manager_redacts_hosted_urls_from_tool_errors():
    secret_url = "https://mcp.example.test/private/tool-token"
    manager = McpManager()
    manager._configured = {"modelscope-arxiv": {"url": secret_url}}
    message = manager.redact_error(RuntimeError(f"POST {secret_url} failed"))
    assert "tool-token" not in message
    assert "https://mcp.example.test/***" in message


def test_product_runtime_shares_mcp_manager_with_research_pipeline(database, tmp_path):
    config = replace(settings, db_path=tmp_path / "test.db", report_dir=tmp_path / "reports")
    runtime = ProductRuntime(database, config)
    assert runtime.pipeline.mcp is runtime.mcp


@pytest.mark.asyncio
async def test_missing_mcp_sdk_is_explicit():
    if importlib.util.find_spec("mcp") is not None:
        pytest.skip("MCP SDK is installed")
    connection = McpConnection("remote", {"type": "streamable-http", "url": "https://example.com/mcp"})
    with pytest.raises(McpDependencyMissing, match="install-mcp"):
        await connection.connect()
