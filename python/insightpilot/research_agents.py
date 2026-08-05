from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .multi_agent import (
    AgentContext,
    AgentResult,
    AgentSpec,
    ResearchState,
    SupervisorDecision,
    ToolRegistry,
)
from .settings import Settings


class ResearchAgent(ABC):
    def __init__(self, spec: AgentSpec, tools: ToolRegistry, settings: Settings):
        self.spec = spec
        self.tools = tools
        self.settings = settings

    @abstractmethod
    async def execute(
        self, state: ResearchState, context: AgentContext
    ) -> AgentResult:
        raise NotImplementedError

    def input_snapshot(self, state: ResearchState) -> dict[str, Any]:
        return {
            "goal": state.goal,
            "search_round": state.search_round,
            "revision_round": state.revision_round,
            "query_count": len(state.queries),
            "search_result_count": len(state.search_results),
            "evidence_count": len(state.evidence),
            "claim_count": len(state.claims),
            "coverage_score": state.coverage_score,
            "private_history_count": 0,
        }


class PlannerAgent(ResearchAgent):
    async def execute(
        self, state: ResearchState, context: AgentContext
    ) -> AgentResult:
        plan = await self.tools.call(
            self.spec.name,
            "plan_research",
            goal=state.goal,
            academic=state.academic,
            gaps=state.gaps,
            conflicts=state.conflicts,
            previous_queries=state.previous_queries + state.queries,
            private_history=context.history_summary(),
        )
        queries = [str(item).strip() for item in plan.get("queries", []) if str(item).strip()]
        if not queries:
            queries = [state.goal]
            plan["queries"] = queries
        return AgentResult(
            agent=self.spec.name,
            status="completed",
            proposed_action="dispatch_research",
            reason="已根据研究目标和上轮证据缺口生成新的检索计划",
            artifacts=[{"kind": "research_plan", "payload": plan}],
            suggested_queries=queries,
            confidence=0.8,
            metrics={"query_count": len(queries)},
        )


class SearchAgent(ResearchAgent):
    def __init__(
        self,
        spec: AgentSpec,
        tools: ToolRegistry,
        settings: Settings,
        tool_name: str,
        provider: str,
    ):
        super().__init__(spec, tools, settings)
        self.tool_name = tool_name
        self.provider = provider

    async def execute(
        self, state: ResearchState, context: AgentContext
    ) -> AgentResult:
        results = await self.tools.call(
            self.spec.name,
            self.tool_name,
            task_id=state.task_id,
            queries=state.queries,
        )
        status = "completed" if results else "needs_input"
        return AgentResult(
            agent=self.spec.name,
            status=status,
            proposed_action="analyze_evidence",
            reason=(
                f"{self.provider} 完成独立检索并返回 {len(results)} 个候选来源"
                if results
                else f"{self.provider} 未返回可用来源"
            ),
            artifacts=[
                {
                    "kind": "search_results",
                    "provider": self.provider,
                    "round": state.search_round,
                    "payload": results,
                }
            ],
            missing_information=[] if results else [f"{self.provider} 没有可用结果"],
            confidence=0.8 if results else 0.2,
            metrics={"result_count": len(results), "provider": self.provider},
        )


class EvidenceAnalystAgent(ResearchAgent):
    async def execute(
        self, state: ResearchState, context: AgentContext
    ) -> AgentResult:
        evidence = await self.tools.call(
            self.spec.name,
            "analyze_sources",
            task_id=state.task_id,
            results=state.search_results,
        )
        return AgentResult(
            agent=self.spec.name,
            status="completed" if evidence else "needs_input",
            proposed_action="verify_claims" if evidence else "search_more",
            reason=(
                f"已从来源中提取并持久化 {len(evidence)} 条可追溯证据"
                if evidence
                else "现有搜索结果无法提取有效证据"
            ),
            artifacts=[{"kind": "evidence_set", "payload": evidence}],
            missing_information=[] if evidence else ["缺少可引用的正文证据"],
            confidence=0.85 if evidence else 0.1,
            metrics={"evidence_count": len(evidence)},
        )


class ClaimVerifierAgent(ResearchAgent):
    async def execute(
        self, state: ResearchState, context: AgentContext
    ) -> AgentResult:
        verification = await self.tools.call(
            self.spec.name,
            "verify_claims",
            task_id=state.task_id,
            goal=state.goal,
            evidence=state.evidence,
            private_history=context.history_summary(),
            recovery_attempt=context.execution_attempt,
            existing_claims=state.claims,
            critique=state.critique,
            revision_round=state.revision_round,
        )
        claims = verification.get("claims", [])
        verified = [claim for claim in claims if claim.get("status") == "verified"]
        unique_sources = len({item.get("source_id") for item in state.evidence if item.get("source_id")})
        citation_score = (
            sum(1 for claim in verified if claim.get("evidence_ids")) / len(verified)
            if verified
            else 0.0
        )
        claim_score = min(len(verified) / max(1, self.settings.min_verified_claims), 1.0)
        source_score = min(unique_sources / max(1, self.settings.min_unique_sources), 1.0)
        coverage = round(0.45 * claim_score + 0.25 * source_score + 0.30 * citation_score, 3)
        conflicts = verification.get("conflicts", [])
        missing = list(verification.get("gaps", []))
        if len(verified) < self.settings.min_verified_claims:
            missing.append(
                f"仅形成 {len(verified)} 个已核验主张，目标至少 {self.settings.min_verified_claims} 个"
            )
        if unique_sources < self.settings.min_unique_sources:
            missing.append(
                f"仅覆盖 {unique_sources} 个独立来源，目标至少 {self.settings.min_unique_sources} 个"
            )
        sufficient = coverage >= self.settings.min_evidence_coverage
        return AgentResult(
            agent=self.spec.name,
            status="completed" if sufficient else "needs_input",
            proposed_action="review_findings" if sufficient else "search_more",
            reason=(
                "主张已通过语义引用核验，冲突将交由 Critic 独立审查"
                if sufficient and conflicts
                else "主张已通过语义引用核验，证据覆盖达到质量门槛"
                if sufficient
                else "证据覆盖或语义核验未达到质量门槛，需要动态返工"
            ),
            artifacts=[
                {
                    "kind": "claim_verification",
                    "payload": {
                        **verification,
                        "coverage_score": coverage,
                        "citation_score": round(citation_score, 3),
                    },
                }
            ],
            missing_information=list(dict.fromkeys(missing)),
            suggested_queries=list(verification.get("suggested_queries", [])),
            confidence=coverage,
            metrics={
                "coverage_score": coverage,
                "citation_score": round(citation_score, 3),
                "verified_claims": len(verified),
                "unique_sources": unique_sources,
                "conflict_count": len(conflicts),
            },
        )


class CriticAgent(ResearchAgent):
    async def execute(
        self, state: ResearchState, context: AgentContext
    ) -> AgentResult:
        critique = await self.tools.call(
            self.spec.name,
            "critique_research",
            task_id=state.task_id,
            goal=state.goal,
            claims=state.claims,
            evidence=state.evidence,
            academic=state.academic,
            coverage_score=state.coverage_score,
            private_history=context.history_summary(),
        )
        decision = str(critique.get("decision", "accept"))
        action = {
            "accept": "write_report",
            "search_more": "search_more",
            "revise_claims": "revise_claims",
            "partial": "write_report",
        }.get(decision, "search_more")
        status = "completed" if decision == "accept" else "needs_input"
        return AgentResult(
            agent=self.spec.name,
            status=status,
            proposed_action=action,
            reason=str(critique.get("reason") or "已完成独立批判性审查"),
            artifacts=[{"kind": "critique", "payload": critique}],
            missing_information=[str(item) for item in critique.get("gaps", [])],
            suggested_queries=[str(item) for item in critique.get("suggested_queries", [])],
            confidence=float(critique.get("overall_confidence", state.coverage_score) or 0),
            metrics={"decision": decision, "risk_count": len(critique.get("risks", []))},
        )


class ReportWriterAgent(ResearchAgent):
    async def execute(
        self, state: ResearchState, context: AgentContext
    ) -> AgentResult:
        report = await self.tools.call(
            self.spec.name,
            "write_report",
            task_id=state.task_id,
            goal=state.goal,
            claims=state.claims,
            evidence=state.evidence,
            critique=state.critique,
            academic=state.academic,
            partial=state.partial,
        )
        verified_count = sum(
            1 for claim in state.claims if claim.get("status", "verified") == "verified"
        )
        return AgentResult(
            agent=self.spec.name,
            status="partial" if state.partial else "completed",
            proposed_action="stop",
            reason=(
                "已仅依据通过核验的主张和证据生成可追溯报告"
                if verified_count
                else "核验未完成，已生成明确区分候选证据与事实结论的部分报告"
            ),
            artifacts=[{"kind": "report", "payload": {"markdown": report}}],
            confidence=state.coverage_score,
            metrics={
                "characters": len(report),
                "partial": state.partial,
                "verified_claims": verified_count,
                "candidate_evidence": len(state.evidence),
            },
        )


class SupervisorAgent:
    """Central coordinator with deterministic safety gates and dynamic routing."""

    spec = AgentSpec(
        name="supervisor",
        objective="根据专业 Agent 的结构化结果动态选择下一行动，并约束循环与失败范围",
        system_prompt="你是研究系统协调者，只通过共享状态和结构化交接进行调度。",
        allowed_tools=frozenset(),
    )

    def __init__(self, settings: Settings):
        self.settings = settings

    def decide(
        self,
        state: ResearchState,
        result: AgentResult,
        *,
        web_available: bool,
        academic_available: bool,
    ) -> SupervisorDecision:
        agent = result.agent
        if result.status == "failed":
            if agent == "planner":
                state.search_round += 1
                state.queries = [state.goal]
                next_agent = "web-researcher" if web_available else "academic-researcher"
                return SupervisorDecision(next_agent, "Planner 连续失败，使用原始目标作为降级检索词")
            if agent == "web-researcher" and academic_available:
                return SupervisorDecision("academic-researcher", "网络检索失败，切换到独立论文检索 Agent")
            if agent == "academic-researcher" and web_available and not state.search_results:
                return SupervisorDecision("web-researcher", "论文检索失败，切换到网络检索 Agent")
            if agent in {"web-researcher", "academic-researcher"} and state.search_results:
                return SupervisorDecision("evidence-analyst", "部分检索失败，使用已获得来源继续")
            if agent == "evidence-analyst" and state.evidence:
                return SupervisorDecision("claim-verifier", "正文分析失败，使用已持久化证据恢复")
            if agent == "claim-verifier":
                state.partial = True
                return SupervisorDecision("report-writer", "核验 Agent 失败，生成明确标注局限的部分报告", partial=True)
            if agent == "critic":
                state.partial = True
                return SupervisorDecision("report-writer", "Critic 失败，保留风险提示并生成部分报告", partial=True)
            return SupervisorDecision("failed", f"{agent} 无可用恢复路径", terminal=True)

        if agent == "planner":
            if web_available:
                return SupervisorDecision("web-researcher", "执行 Planner 提交的网络检索计划")
            return SupervisorDecision("academic-researcher", "仅论文检索可用，派发 Academic Researcher")

        if agent == "web-researcher":
            if academic_available and state.completed_research_rounds.get("academic-researcher") != state.search_round:
                return SupervisorDecision("academic-researcher", "独立论文研究角色尚未完成本轮检索")
            return SupervisorDecision("evidence-analyst", "本轮研究 Agent 已完成，交接来源分析")

        if agent == "academic-researcher":
            if web_available and state.completed_research_rounds.get("web-researcher") != state.search_round:
                return SupervisorDecision("web-researcher", "独立网络研究角色尚未完成本轮检索")
            return SupervisorDecision("evidence-analyst", "本轮研究 Agent 已完成，交接来源分析")

        if agent == "evidence-analyst":
            if result.status == "needs_input":
                if state.search_round < self.settings.max_search_rounds:
                    return SupervisorDecision("planner", "没有提取到有效证据，要求 Planner 改写查询")
                state.partial = True
            return SupervisorDecision("claim-verifier", "将独立证据产物交接给 Claim Verifier", partial=state.partial)

        if agent == "claim-verifier":
            if result.proposed_action == "search_more":
                if state.search_round < self.settings.max_search_rounds:
                    return SupervisorDecision("planner", "证据质量门槛未通过，触发补充检索")
                state.partial = True
                return SupervisorDecision("critic", "达到检索轮次上限，交由 Critic 决定结论边界", partial=True)
            return SupervisorDecision("critic", "证据质量门槛通过，进入独立批判审查")

        if agent == "critic":
            if result.proposed_action == "search_more":
                if state.search_round < self.settings.max_search_rounds:
                    return SupervisorDecision("planner", "Critic 发现证据缺口，触发定向补检索")
                state.partial = True
                return SupervisorDecision("report-writer", "检索预算已耗尽，生成部分报告", partial=True)
            if result.proposed_action == "revise_claims":
                if state.revision_round < self.settings.max_revision_rounds:
                    state.revision_round += 1
                    return SupervisorDecision("claim-verifier", "Critic 驳回部分主张，返回 Verifier 修订")
                state.partial = True
                return SupervisorDecision("report-writer", "主张修订达到上限，生成部分报告", partial=True)
            if str(result.metrics.get("decision")) == "partial":
                state.partial = True
            return SupervisorDecision("report-writer", "Critic 已确定报告边界，交接 Writer", partial=state.partial)

        if agent == "report-writer":
            return SupervisorDecision("completed", "最终报告已生成", terminal=True, partial=state.partial)

        return SupervisorDecision("failed", f"未知 Agent：{agent}", terminal=True)


def default_agent_specs() -> dict[str, AgentSpec]:
    return {
        "planner": AgentSpec(
            "planner",
            "拆解研究目标，依据证据缺口生成互补且不重复的检索计划",
            "你是独立研究规划 Agent，不执行检索，只输出结构化研究计划。",
            frozenset({"plan_research"}),
        ),
        "web-researcher": AgentSpec(
            "web-researcher",
            "通过实时网络检索获取多样、近期且可追溯的来源",
            "你是网络研究 Agent，只能使用网络检索工具，不验证主张或撰写报告。",
            frozenset({"web_search"}),
        ),
        "academic-researcher": AgentSpec(
            "academic-researcher",
            "通过 arXiv MCP 获取可追溯论文来源",
            "你是论文研究 Agent，只能使用学术检索工具。",
            frozenset({"academic_search"}),
        ),
        "evidence-analyst": AgentSpec(
            "evidence-analyst",
            "抓取来源正文并提取带出处的证据片段",
            "你是证据分析 Agent，只处理来源，不生成最终主张。",
            frozenset({"analyze_sources"}),
        ),
        "claim-verifier": AgentSpec(
            "claim-verifier",
            "验证主张与证据之间的语义支持、冲突和不足关系",
            "你是主张核验 Agent，只能保存具有精确原文支持的主张。",
            frozenset({"verify_claims"}),
        ),
        "critic": AgentSpec(
            "critic",
            "独立审查证据覆盖、来源偏差、冲突和结论边界",
            "你是独立 Critic，可以驳回上游结果并要求补检索或修订主张。",
            frozenset({"critique_research"}),
        ),
        "report-writer": AgentSpec(
            "report-writer",
            "仅依据已核验主张生成带真实引用的报告",
            "你是报告 Agent，不能搜索或修改证据，只能读取核验结果。",
            frozenset({"write_report"}),
        ),
    }
