from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import replace

import pytest

from insightpilot.database import Database, utcnow
from insightpilot.multi_agent import (
    AgentContext,
    AgentResult,
    AgentSpec,
    ResearchState,
    ToolPermissionError,
    ToolRegistry,
)
from insightpilot.research_agents import SupervisorAgent
from insightpilot.research_pipeline import ResearchPipeline, sha256
from insightpilot.providers import ProviderUnavailable
from insightpilot.settings import settings


@pytest.fixture()
def database(tmp_path):
    db = Database(tmp_path / "multi-agent.db")
    db.init()
    return db


def add_verification_evidence(database, task_id, evidence_ids):
    for evidence_id in evidence_ids:
        source_id = f"source-{evidence_id}"
        quote = f"Official source {evidence_id} contains an exact verifiable statement for testing."
        database.insert(
            "sources",
            {
                "id": source_id,
                "task_id": task_id,
                "url": f"https://example.gov/jobs/{evidence_id}",
                "title": f"Source {evidence_id}",
                "published_at": None,
                "retrieved_at": utcnow(),
                "source_type": "official",
                "fetch_status": "fetched",
                "content_hash": sha256(source_id),
                "raw_text": quote,
            },
        )
        database.insert(
            "evidence",
            {
                "id": evidence_id,
                "task_id": task_id,
                "source_id": source_id,
                "claim": None,
                "quote": quote,
                "reliability": 0.95,
                "stance": "supports",
                "content_hash": sha256(evidence_id),
                "created_at": utcnow(),
            },
        )


def verification_payload(batch):
    return {
        "claims": [
            {
                "text": f"Verified finding for {item['evidence_id']}",
                "confidence": 0.9,
                "evidence": [
                    {
                        "evidence_id": item["evidence_id"],
                        "relation": "entailed",
                        "entailment_score": 0.9,
                        "supporting_quote": item["quote"],
                        "reason": "exact",
                    }
                ],
            }
            for item in batch
        ],
        "gaps": [],
        "conflicts": [],
        "suggested_queries": [],
    }


def test_replace_verified_claims_is_atomic(database):
    task_id = database.create_task("atomic claim replacement")
    add_verification_evidence(database, task_id, ["e1"])
    database.insert(
        "claims",
        {
            "id": "old-claim",
            "task_id": task_id,
            "text": "Existing valid claim",
            "confidence": 0.8,
            "status": "verified",
            "created_at": utcnow(),
        },
    )
    database.save_verification_batch(
        task_id,
        "batch-key",
        ["missing-evidence"],
        "completed",
        "test",
        "claim-verifier-v2",
        payload={"claims": []},
    )

    with pytest.raises(sqlite3.IntegrityError):
        database.replace_verified_claims(
            task_id,
            [{
                "claim_id": "new-claim",
                "text": "Invalid replacement claim",
                "confidence": 0.9,
                "status": "verified",
                "verifications": [{
                    "evidence_id": "missing-evidence",
                    "relation": "entailed",
                    "entailment_score": 0.9,
                    "supporting_quote": "missing",
                    "reason": "invalid foreign key",
                }],
            }],
            ["batch-key"],
        )

    assert database.fetchone("SELECT id FROM claims WHERE id='old-claim'") is not None
    assert database.fetchone("SELECT id FROM claims WHERE id='new-claim'") is None
    batch = database.fetchone(
        "SELECT applied FROM verification_batches WHERE task_id=? AND cache_key=?",
        (task_id, "batch-key"),
    )
    assert batch["applied"] == 0


@pytest.mark.asyncio
async def test_tool_registry_enforces_agent_permissions():
    registry = ToolRegistry()
    registry.register("web_search", lambda query: [query])
    web = AgentSpec("web", "search", "web", frozenset({"web_search"}))
    writer = AgentSpec("writer", "write", "writer", frozenset())
    registry.grant(web)
    registry.grant(writer)

    assert await registry.call("web", "web_search", query="agents") == ["agents"]
    with pytest.raises(ToolPermissionError):
        await registry.call("writer", "web_search", query="forbidden")


def test_supervisor_routes_verifier_back_to_planner_until_budget():
    config = replace(settings, max_search_rounds=2, min_evidence_coverage=0.72)
    supervisor = SupervisorAgent(config)
    state = ResearchState("task", "goal", academic=False, search_round=1)
    handoff = AgentResult(
        agent="claim-verifier",
        status="needs_input",
        proposed_action="search_more",
        reason="coverage low",
        confidence=0.4,
    )

    decision = supervisor.decide(
        state, handoff, web_available=True, academic_available=False
    )
    assert decision.next_agent == "planner"

    state.search_round = 2
    decision = supervisor.decide(
        state, handoff, web_available=True, academic_available=False
    )
    assert decision.next_agent == "critic"
    assert decision.partial is False
    assert state.partial is False


def quality_state(*, search_round=3, stagnant_rounds=0):
    evidence = [
        {"evidence_id": f"e{index}", "source_id": f"s{index}"}
        for index in range(1, 4)
    ]
    claims = [
        {
            "text": f"Verified claim {index}",
            "status": "verified",
            "evidence_ids": [f"e{index}"],
        }
        for index in range(1, 4)
    ]
    return ResearchState(
        "task",
        "goal",
        academic=False,
        search_round=search_round,
        stagnant_rounds=stagnant_rounds,
        evidence=evidence,
        claims=claims,
        coverage_score=1.0,
        citation_score=1.0,
    )


def test_supervisor_completes_at_budget_when_only_advisory_gaps_remain():
    config = replace(settings, max_search_rounds=3)
    supervisor = SupervisorAgent(config)
    state = quality_state()
    state.critique = {
        "decision": "search_more",
        "gaps": ["More examples would improve breadth"],
        "blocking_gaps": [],
        "advisory_gaps": ["More examples would improve breadth"],
    }
    result = AgentResult(
        "critic",
        "needs_input",
        "search_more",
        "advisory gap",
        metrics={"decision": "search_more"},
    )

    decision = supervisor.decide(
        state, result, web_available=True, academic_available=False
    )

    assert decision.next_agent == "report-writer"
    assert decision.partial is False
    assert state.partial is False
    assert "More examples would improve breadth" in state.limitations


def test_supervisor_stops_no_progress_loop_before_search_budget():
    config = replace(settings, max_search_rounds=3)
    supervisor = SupervisorAgent(config)
    state = quality_state(search_round=2, stagnant_rounds=1)
    state.critique = {
        "decision": "search_more",
        "gaps": ["Optional broader sample"],
        "blocking_gaps": [],
    }
    result = AgentResult(
        "critic",
        "needs_input",
        "search_more",
        "no material progress",
        metrics={"decision": "search_more"},
    )

    decision = supervisor.decide(
        state, result, web_available=True, academic_available=False
    )

    assert decision.next_agent == "report-writer"
    assert decision.partial is False
    assert "补充检索没有新增有效覆盖" in state.limitations


def test_supervisor_keeps_partial_for_missing_required_dimension():
    config = replace(settings, max_search_rounds=3)
    supervisor = SupervisorAgent(config)
    state = quality_state()
    state.critique = {
        "decision": "search_more",
        "gaps": ["No direct evidence for LLM-based airfoil design"],
        "blocking_gaps": ["No direct evidence for LLM-based airfoil design"],
    }
    result = AgentResult(
        "critic",
        "needs_input",
        "search_more",
        "core objective missing",
        metrics={"decision": "search_more"},
    )

    decision = supervisor.decide(
        state, result, web_available=True, academic_available=False
    )

    assert decision.next_agent == "report-writer"
    assert decision.partial is True
    assert state.partial_reason
    assert "LLM-based airfoil design" in state.partial_reason


@pytest.mark.asyncio
async def test_critic_cannot_expand_scope_after_quality_gate_passes(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
    )
    pipeline = ResearchPipeline(database, config)

    async def expanded_scope_critique(**kwargs):
        return {
            "decision": "search_more",
            "reason": "Add optional interview and salary sections",
            "gaps": ["Interview process", "Salary history"],
            "blocking_gaps": [],
            "advisory_gaps": ["Interview process", "Salary history"],
            "risks": [],
            "suggested_queries": ["salary history"],
            "overall_confidence": 0.9,
        }

    pipeline.tools._tools["critique_research"] = expanded_scope_critique
    state = quality_state()
    state.required_dimensions = ["Job responsibilities", "Core skills"]
    agent = pipeline.agents["critic"]

    result = await agent.execute(state, AgentContext("task", agent.spec))

    assert result.status == "completed"
    assert result.proposed_action == "write_report"
    assert result.metrics["decision"] == "accept"


@pytest.mark.asyncio
async def test_planner_aspects_cannot_expand_blocking_contract(database, tmp_path):
    pipeline = ResearchPipeline(database, replace(settings, db_path=tmp_path / "db.sqlite"))

    async def expanded_plan(*args, **kwargs):
        return {
            "queries": ["company jobs"],
            "aspects": ["Job categories", "Future hiring cycle", "Salary"],
            "required_dimensions": ["Future hiring cycle", "Salary"],
            "optional_dimensions": [],
            "completion_criteria": [],
        }

    pipeline.llm.json = expanded_plan
    goal = "Research the company's job openings"
    plan = await pipeline._create_plan(goal)

    assert plan["required_dimensions"] == [goal]
    assert "Future hiring cycle" in plan["optional_dimensions"]
    assert "Salary" in plan["optional_dimensions"]


@pytest.mark.asyncio
async def test_critic_demotes_blocker_not_aligned_to_user_goal(database, tmp_path):
    pipeline = ResearchPipeline(database, replace(settings, db_path=tmp_path / "db.sqlite"))
    task_id = database.create_task("Research company jobs")

    async def expanded_critique(*args, **kwargs):
        return {
            "decision": "search_more",
            "reason": "Salary is unavailable",
            "gaps": [],
            "blocking_gaps": [
                {"dimension": "Salary", "description": "No salary history"}
            ],
            "advisory_gaps": [],
            "covered_dimensions": [],
            "risks": [],
            "suggested_queries": [],
            "overall_confidence": 0.8,
        }

    pipeline.llm.json = expanded_critique
    critique = await pipeline._critique(
        task_id,
        "Research company jobs",
        [],
        [],
        coverage_score=1.0,
        required_dimensions=["Research company jobs"],
    )

    assert critique["blocking_gaps"] == []
    assert critique["blocking_dimensions"] == []
    assert critique["advisory_gaps"] == ["No salary history"]


def test_repeated_core_gap_marks_research_as_stagnant(database, tmp_path):
    pipeline = ResearchPipeline(database, replace(settings, db_path=tmp_path / "db.sqlite"))
    state = ResearchState("task", "goal", academic=False)
    critique = AgentResult(
        "critic",
        "needs_input",
        "search_more",
        "same blocker",
        artifacts=[{
            "kind": "critique",
            "payload": {
                "decision": "search_more",
                "blocking_gaps": ["Missing core evidence"],
                "blocking_dimensions": ["goal"],
            },
        }],
    )

    pipeline._apply_agent_result(state, critique)
    pipeline._apply_agent_result(state, critique)

    assert state.stagnant_core_rounds == 1


def test_report_citation_gate_allows_omitted_non_core_claims():
    known_id = "11111111-1111-1111-1111-111111111111"
    omitted_id = "22222222-2222-2222-2222-222222222222"
    claims = [
        {"text": "Included", "evidence_ids": [known_id]},
        {"text": "Not expanded in report", "evidence_ids": [omitted_id]},
    ]
    report = f"Finding [{known_id}]\n\nhttps://example.com/source"

    assert ResearchPipeline._report_citations_valid(
        report,
        claims,
        {known_id, omitted_id},
        {"https://example.com/source"},
    )


@pytest.mark.parametrize(
    "report",
    [
        "Unknown evidence [99999999-9999-9999-9999-999999999999]",
        "Known [11111111-1111-1111-1111-111111111111] https://invented.example/source",
        "No evidence citation at all",
    ],
)
def test_report_citation_gate_rejects_hard_integrity_errors(report):
    known_id = "11111111-1111-1111-1111-111111111111"
    claims = [{"text": "Included", "evidence_ids": [known_id]}]

    assert not ResearchPipeline._report_citations_valid(
        report,
        claims,
        {known_id},
        {"https://example.com/source"},
    )


@pytest.mark.asyncio
async def test_verifier_sends_conflicts_to_critic_when_coverage_is_sufficient(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        min_verified_claims=1,
        min_unique_sources=1,
        min_evidence_coverage=0.7,
    )
    pipeline = ResearchPipeline(database, config)

    async def fake_verify(**kwargs):
        return {
            "claims": [{
                "text": "Supported finding",
                "status": "verified",
                "evidence_ids": ["e1"],
            }],
            "gaps": [],
            "conflicts": ["A separate candidate claim is contradicted"],
            "suggested_queries": [],
        }

    pipeline.tools._tools["verify_claims"] = fake_verify
    state = ResearchState(
        "task",
        "goal",
        academic=False,
        evidence=[{"evidence_id": "e1", "source_id": "s1"}],
    )
    agent = pipeline.agents["claim-verifier"]
    result = await agent.execute(state, AgentContext("task", agent.spec))
    assert result.status == "completed"
    assert result.proposed_action == "review_findings"
    assert result.metrics["conflict_count"] == 1


def test_database_persists_private_context_checkpoint_and_trace(database):
    task_id = database.create_task("trace agents")
    database.add_agent_message(task_id, "planner", "input", "private plan")
    database.add_agent_message(task_id, "critic", "input", "private critique")
    database.save_checkpoint(
        task_id,
        "web-researcher",
        ResearchState(task_id, "trace agents", academic=False).to_dict(),
    )
    run_id = database.start_agent_run(
        task_id, "planner", "plan", ["plan_research"], 0, {"goal": "trace agents"}
    )
    database.finish_agent_run(run_id, "completed", {"next_agent": "web-researcher"})

    assert database.agent_messages(task_id, "planner")[0]["content"] == "private plan"
    assert database.agent_messages(task_id, "critic")[0]["content"] == "private critique"
    trace = database.agent_trace(task_id)
    assert trace["checkpoint"]["current_agent"] == "web-researcher"
    assert trace["runs"][0]["allowed_tools"] == ["plan_research"]


@pytest.mark.asyncio
async def test_claim_verifier_requires_exact_supporting_quote(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("verify evidence")
    source_id = "source-1"
    evidence_id = "evidence-1"
    quote = "The benchmark reports a 12 percent improvement on the public test set."
    database.insert(
        "sources",
        {
            "id": source_id,
            "task_id": task_id,
            "url": "https://example.com/paper",
            "title": "Paper",
            "published_at": None,
            "retrieved_at": utcnow(),
            "source_type": "paper",
            "fetch_status": "fetched",
            "content_hash": sha256(quote),
            "raw_text": quote,
        },
    )
    database.insert(
        "evidence",
        {
            "id": evidence_id,
            "task_id": task_id,
            "source_id": source_id,
            "claim": None,
            "quote": quote,
            "reliability": 0.9,
            "stance": "supports",
            "content_hash": sha256("evidence"),
            "created_at": utcnow(),
        },
    )

    async def fake_json(*args, **kwargs):
        return {
            "claims": [
                {
                    "text": "The benchmark improved by 12 percent.",
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "evidence_id": evidence_id,
                            "relation": "entailed",
                            "entailment_score": 0.92,
                            "supporting_quote": quote,
                            "reason": "direct statement",
                        }
                    ],
                },
                {
                    "text": "The benchmark improved by 50 percent.",
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "evidence_id": evidence_id,
                            "relation": "entailed",
                            "entailment_score": 0.99,
                            "supporting_quote": "This quote does not exist in the source.",
                            "reason": "fabricated",
                        }
                    ],
                },
            ],
            "gaps": [],
            "conflicts": [],
            "suggested_queries": [],
        }

    pipeline.llm.json = fake_json
    result = await pipeline._verify_claims_tool(
        task_id, "verify evidence", pipeline._load_evidence(task_id)
    )

    assert [claim["text"] for claim in result["claims"]] == [
        "The benchmark improved by 12 percent."
    ]
    verification = database.fetchone(
        "SELECT * FROM claim_verifications WHERE evidence_id=?", (evidence_id,)
    )
    assert verification["relation"] == "entailed"
    assert verification["supporting_quote"] == quote


@pytest.mark.asyncio
async def test_claim_verifier_splits_only_failed_batch(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
        verifier_batch_size=4,
        max_verification_evidence=4,
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("verify batched evidence")
    evidence = []
    for index in range(4):
        source_id = f"source-{index}"
        evidence_id = f"evidence-{index}"
        quote = f"Source {index} states a verifiable finding with enough exact quotation text."
        database.insert("sources", {
            "id": source_id, "task_id": task_id, "url": f"https://example.com/{index}",
            "title": f"Source {index}", "published_at": None, "retrieved_at": utcnow(),
            "source_type": "web", "fetch_status": "fetched", "content_hash": sha256(source_id),
            "raw_text": quote,
        })
        database.insert("evidence", {
            "id": evidence_id, "task_id": task_id, "source_id": source_id, "claim": None,
            "quote": quote, "reliability": 0.8, "stance": "supports",
            "content_hash": sha256(evidence_id), "created_at": utcnow(),
        })
        evidence.append({
            "evidence_id": evidence_id, "source_id": source_id,
            "url": f"https://example.com/{index}", "title": f"Source {index}",
            "quote": quote, "reliability": 0.8, "fetch_status": "fetched",
            "created_at": utcnow(), "retrieved_at": utcnow(),
        })

    calls = 0

    async def fake_json(system, user, **kwargs):
        nonlocal calls
        calls += 1
        match = re.search(r"本批证据：(\[.*\])\nVerifier", user, re.S)
        batch = json.loads(match.group(1))
        if len(batch) == 4:
            raise ProviderUnavailable("batch too large")
        item = batch[0]
        return {
            "claims": [{
                "text": f"Verified {item['id']}", "confidence": 0.9,
                "evidence": [{
                    "evidence_id": item["id"], "relation": "entailed",
                    "entailment_score": 0.9, "supporting_quote": item["quote"],
                    "reason": "exact",
                }],
            }],
            "gaps": [], "conflicts": [], "suggested_queries": [],
        }

    pipeline.llm.json = fake_json
    result = await pipeline._verify_claims_tool(task_id, "verify batched evidence", evidence)
    assert calls == 3
    assert len(result["claims"]) == 2
    split = database.fetchone(
        "SELECT id FROM events WHERE task_id=? AND event_type='verification.batch_split'",
        (task_id,),
    )
    assert split is not None


@pytest.mark.asyncio
async def test_claim_verifier_deduplicates_claim_evidence_pairs(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
        max_verification_evidence=1,
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("deduplicate verification links")
    add_verification_evidence(database, task_id, ["e1"])
    evidence = pipeline._load_evidence(task_id)

    async def duplicate_batch(goal, batch, private_history):
        item = batch[0]
        return {
            "claims": [{
                "text": "A conservatively resolved finding",
                "confidence": 0.9,
                "evidence": [
                    {
                        "evidence_id": item["evidence_id"],
                        "relation": "entailed",
                        "entailment_score": 0.95,
                        "supporting_quote": item["quote"],
                        "reason": "supports",
                    },
                    {
                        "evidence_id": item["evidence_id"],
                        "relation": "contradicted",
                        "entailment_score": 0.7,
                        "supporting_quote": item["quote"],
                        "reason": "conflicts",
                    },
                ],
            }],
            "gaps": [], "conflicts": [], "suggested_queries": [],
        }

    pipeline._verify_claim_batch = duplicate_batch
    result = await pipeline._verify_claims_tool(task_id, "deduplicate", evidence)

    assert result["claims"][0]["status"] == "mixed"
    rows = database.fetchall(
        "SELECT * FROM claim_verifications WHERE claim_id=?",
        (result["claims"][0]["claim_id"],),
    )
    assert len(rows) == 1
    assert rows[0]["relation"] == "contradicted"


@pytest.mark.asyncio
async def test_claim_verifier_only_processes_new_evidence_across_rounds(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
        verifier_batch_size=2,
        max_verification_evidence=6,
        followup_verification_evidence=2,
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("incremental verification")
    add_verification_evidence(database, task_id, ["e1", "e2", "e3", "e4"])
    calls = []

    async def record_batch(goal, batch, private_history):
        calls.append({item["evidence_id"] for item in batch})
        return verification_payload(batch)

    pipeline._verify_claim_batch = record_batch
    first = await pipeline._verify_claims_tool(
        task_id, "incremental verification", pipeline._load_evidence(task_id)
    )
    assert len(calls) == 2

    add_verification_evidence(database, task_id, ["e5", "e6", "e7", "e8"])
    second = await pipeline._verify_claims_tool(
        task_id,
        "incremental verification",
        pipeline._load_evidence(task_id),
        existing_claims=first["claims"],
    )

    assert len(calls) == 3
    assert len(calls[-1]) == 2
    assert calls[-1].issubset({"e5", "e6", "e7", "e8"})
    assert len(second["claims"]) == 6
    assert len(database.completed_verification_evidence(
        task_id, "test", "claim-verifier-v2"
    )) == 6


@pytest.mark.asyncio
async def test_claim_verifier_retries_only_failed_batch(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
        verifier_batch_size=2,
        max_verification_evidence=4,
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("resume failed verification batch")
    add_verification_evidence(database, task_id, ["e1", "e2", "e3", "e4"])
    calls = []
    failed_batch = set()

    async def fail_once(goal, batch, private_history):
        ids = {item["evidence_id"] for item in batch}
        calls.append(ids)
        if not failed_batch:
            failed_batch.update(ids)
            raise ProviderUnavailable("temporary batch failure")
        return verification_payload(batch)

    pipeline._verify_claim_batch = fail_once
    first = await pipeline._verify_claims_tool(
        task_id, "resume failed batch", pipeline._load_evidence(task_id)
    )
    assert len(calls) == 2

    second = await pipeline._verify_claims_tool(
        task_id,
        "resume failed batch",
        pipeline._load_evidence(task_id),
        existing_claims=first["claims"],
    )

    assert len(calls) == 3
    assert calls[-1] == failed_batch
    assert len(second["claims"]) == 4


@pytest.mark.asyncio
async def test_claim_verifier_reuses_checkpoint_after_commit_failure(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
        max_verification_evidence=1,
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("resume after commit failure")
    add_verification_evidence(database, task_id, ["e1"])
    calls = 0

    async def record_batch(goal, batch, private_history):
        nonlocal calls
        calls += 1
        return verification_payload(batch)

    pipeline._verify_claim_batch = record_batch
    original_replace = database.replace_verified_claims

    def fail_commit(*args, **kwargs):
        raise RuntimeError("simulated commit failure")

    database.replace_verified_claims = fail_commit
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        await pipeline._verify_claims_tool(
            task_id, "resume commit", pipeline._load_evidence(task_id)
        )
    assert calls == 1

    database.replace_verified_claims = original_replace
    result = await pipeline._verify_claims_tool(
        task_id, "resume commit", pipeline._load_evidence(task_id)
    )

    assert calls == 1
    assert len(result["claims"]) == 1
    assert database.unapplied_verification_batches(
        task_id, "test", "claim-verifier-v2"
    ) == []


@pytest.mark.asyncio
async def test_verifier_revision_removes_only_critic_rejected_claims(database, tmp_path):
    config = replace(settings, db_path=tmp_path / "multi-agent.db", report_dir=tmp_path / "reports")
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("revise rejected claims")
    database.insert("sources", {
        "id": "s1", "task_id": task_id, "url": "https://example.com/source",
        "title": "Source", "published_at": None, "retrieved_at": utcnow(),
        "source_type": "web", "fetch_status": "fetched", "content_hash": sha256("source"),
        "raw_text": "Exact evidence text that is sufficiently long for claim verification.",
    })
    database.insert("evidence", {
        "id": "e1", "task_id": task_id, "source_id": "s1", "claim": None,
        "quote": "Exact evidence text that is sufficiently long for claim verification.",
        "reliability": 0.8, "stance": "supports", "content_hash": sha256("e1"),
        "created_at": utcnow(),
    })
    for claim_id, text in (("keep-claim-0001", "Keep this claim"), ("drop-claim-0002", "Drop this claim")):
        database.insert("claims", {
            "id": claim_id, "task_id": task_id, "text": text,
            "confidence": 0.9, "status": "verified", "created_at": utcnow(),
        })
        database.insert("claim_evidence", {
            "claim_id": claim_id, "evidence_id": "e1", "relation": "supports",
        })

    async def unexpected_json(*args, **kwargs):
        raise AssertionError("targeted revision should not re-run full evidence verification")

    pipeline.llm.json = unexpected_json
    result = await pipeline._verify_claims_tool(
        task_id,
        "revise rejected claims",
        pipeline._load_evidence(task_id),
        existing_claims=pipeline._load_claims(task_id),
        critique={"gaps": ["drop-cla has weak support"], "suggested_queries": []},
        revision_round=1,
    )
    assert [claim["claim_id"] for claim in result["claims"]] == ["keep-claim-0001"]
    assert database.fetchone("SELECT id FROM claims WHERE id='drop-claim-0002'") is None


def test_verifier_selection_keeps_new_high_quality_source(database, tmp_path):
    config = replace(settings, max_verification_evidence=6)
    pipeline = ResearchPipeline(database, config)
    old = [{
        "evidence_id": f"old-{index}", "source_id": "community-source",
        "url": "https://blog.csdn.net/example/article/details/1", "quote": "old evidence",
        "reliability": 0.8, "fetch_status": "fetched", "created_at": f"2026-01-{index + 1:02d}",
    } for index in range(20)]
    new = {
        "evidence_id": "new-official", "source_id": "official-source",
        "url": "https://example.gov.cn/jobs/llm", "quote": "new official evidence",
        "reliability": 0.9, "fetch_status": "fetched", "created_at": "2026-08-05",
    }
    selected = pipeline._select_verification_evidence(old + [new])
    assert "new-official" in {item["evidence_id"] for item in selected}


def test_terminal_checkpoint_resumes_failed_verifier(database, tmp_path):
    pipeline = ResearchPipeline(database, replace(settings, llm_api_key="test", llm_model="test"))
    state = ResearchState(
        "task", "goal", academic=False, next_agent="completed",
        evidence=[{"evidence_id": "e1"}],
        errors=[{"agent": "claim-verifier", "error": "empty response"}],
    )
    assert pipeline._resume_agent(state) == "claim-verifier"


class StubAgent:
    def __init__(self, spec, handler):
        self.spec = spec
        self.handler = handler

    def input_snapshot(self, state):
        return {"goal": state.goal, "search_round": state.search_round}

    async def execute(self, state, context):
        return self.handler(state)


@pytest.mark.asyncio
async def test_resume_quality_gate_skips_redundant_verification(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
        tavily_api_key="test",
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("resume verified research")
    database.insert("sources", {
        "id": "s1", "task_id": task_id, "url": "https://example.com/source",
        "title": "Source", "published_at": None, "retrieved_at": utcnow(),
        "source_type": "web", "fetch_status": "fetched", "content_hash": sha256("source"),
        "raw_text": "A sufficiently long exact source quotation for checkpoint recovery.",
    })
    database.insert("evidence", {
        "id": "e1", "task_id": task_id, "source_id": "s1", "claim": None,
        "quote": "A sufficiently long exact source quotation for checkpoint recovery.",
        "reliability": 0.8, "stance": "supports", "content_hash": sha256("e1"),
        "created_at": utcnow(),
    })
    database.insert("claims", {
        "id": "c1", "task_id": task_id, "text": "Verified checkpoint finding",
        "confidence": 0.9, "status": "verified", "created_at": utcnow(),
    })
    database.insert("claim_evidence", {
        "claim_id": "c1", "evidence_id": "e1", "relation": "supports",
    })
    state = ResearchState(
        task_id, "resume verified research", academic=False,
        next_agent="claim-verifier", coverage_score=1.0, citation_score=1.0,
    )
    database.save_checkpoint(task_id, "claim-verifier", state.to_dict())
    database.update_task(task_id, "cancelled")

    pipeline.agents["critic"] = StubAgent(
        pipeline.agent_specs["critic"],
        lambda state: AgentResult(
            "critic", "completed", "write_report", "accepted",
            artifacts=[{"kind": "critique", "payload": {"decision": "accept", "gaps": [], "risks": []}}],
            metrics={"decision": "accept"},
        ),
    )
    pipeline.agents["report-writer"] = StubAgent(
        pipeline.agent_specs["report-writer"],
        lambda state: AgentResult(
            "report-writer", "completed", "stop", "written",
            artifacts=[{"kind": "report", "payload": {"markdown": "# Report\n\nVerified checkpoint finding [e1]"}}],
        ),
    )

    result = await pipeline.run(task_id)
    specialist_runs = [
        run["agent"] for run in database.agent_trace(task_id)["runs"]
        if run["agent"] != "supervisor"
    ]
    assert specialist_runs == ["critic", "report-writer"]
    assert result["partial"] is False


@pytest.mark.asyncio
async def test_full_workflow_dynamically_replans_after_verifier_rejection(database, tmp_path):
    config = replace(
        settings,
        db_path=tmp_path / "multi-agent.db",
        report_dir=tmp_path / "reports",
        llm_api_key="test",
        llm_model="test",
        tavily_api_key="test",
        max_search_rounds=2,
    )
    pipeline = ResearchPipeline(database, config)
    task_id = database.create_task("research a changing market")

    def planner(state):
        round_number = state.search_round + 1
        return AgentResult(
            "planner",
            "completed",
            "dispatch_research",
            "plan ready",
            artifacts=[
                {
                    "kind": "research_plan",
                    "payload": {"queries": [f"market evidence round {round_number}"]},
                }
            ],
        )

    def web(state):
        assert state.claims == []
        assert state.report == ""
        return AgentResult(
            "web-researcher",
            "completed",
            "analyze_evidence",
            "sources ready",
            artifacts=[
                {
                    "kind": "search_results",
                    "provider": "stub-web",
                    "payload": [{"url": f"https://example.com/{state.search_round}"}],
                }
            ],
        )

    def analyst(state):
        evidence = [
            {
                "evidence_id": f"e{state.search_round}",
                "source_id": f"s{state.search_round}",
                "url": f"https://example.com/{state.search_round}",
                "title": "Source",
                "retrieved_at": "now",
                "quote": "Attributable evidence",
                "reliability": 0.8,
            }
        ]
        return AgentResult(
            "evidence-analyst",
            "completed",
            "verify_claims",
            "evidence ready",
            artifacts=[{"kind": "evidence_set", "payload": evidence}],
        )

    def verifier(state):
        enough = state.search_round >= 2
        return AgentResult(
            "claim-verifier",
            "completed" if enough else "needs_input",
            "review_findings" if enough else "search_more",
            "coverage sufficient" if enough else "coverage insufficient",
            artifacts=[
                {
                    "kind": "claim_verification",
                    "payload": {
                        "claims": [
                            {
                                "text": "Verified result",
                                "status": "verified",
                                "evidence_ids": [f"e{state.search_round}"],
                            }
                        ],
                        "coverage_score": 0.9 if enough else 0.4,
                        "citation_score": 1.0,
                        "conflicts": [],
                    },
                }
            ],
            missing_information=[] if enough else ["need another independent source"],
        )

    def critic(state):
        return AgentResult(
            "critic",
            "completed",
            "write_report",
            "accepted",
            artifacts=[
                {
                    "kind": "critique",
                    "payload": {"decision": "accept", "gaps": [], "risks": []},
                }
            ],
            metrics={"decision": "accept"},
        )

    def writer(state):
        assert state.queries == []
        assert state.search_results == []
        return AgentResult(
            "report-writer",
            "completed",
            "stop",
            "report ready",
            artifacts=[
                {
                    "kind": "report",
                    "payload": {"markdown": "# Report\n\nVerified result [e2]"},
                }
            ],
        )

    handlers = {
        "planner": planner,
        "web-researcher": web,
        "evidence-analyst": analyst,
        "claim-verifier": verifier,
        "critic": critic,
        "report-writer": writer,
    }
    for name, handler in handlers.items():
        pipeline.agents[name] = StubAgent(pipeline.agent_specs[name], handler)

    result = await pipeline.run(task_id)
    trace = database.agent_trace(task_id)
    specialist_names = [run["agent"] for run in trace["runs"] if run["agent"] != "supervisor"]

    assert result["search_rounds"] == 2
    assert specialist_names.count("planner") == 2
    assert specialist_names.count("claim-verifier") == 2
    assert result["partial"] is False
    assert database.task(task_id)["status"] == "completed"
    assert trace["checkpoint"]["current_agent"] == "completed"
