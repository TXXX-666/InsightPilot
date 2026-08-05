from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Literal


AgentStatus = Literal["completed", "needs_input", "failed", "partial"]
AgentAction = Literal[
    "dispatch_research",
    "analyze_evidence",
    "verify_claims",
    "review_findings",
    "search_more",
    "revise_claims",
    "write_report",
    "stop",
]


class ToolPermissionError(PermissionError):
    pass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    objective: str
    system_prompt: str
    allowed_tools: frozenset[str] = frozenset()


@dataclass
class AgentContext:
    task_id: str
    agent: AgentSpec
    private_messages: list[dict[str, Any]] = field(default_factory=list)
    execution_attempt: int = 1

    def history_summary(self, limit: int = 4) -> str:
        rows = self.private_messages[-limit:]
        if not rows:
            return "无历史执行记录。"
        return "\n".join(
            f"- {row.get('role', 'unknown')}: {str(row.get('content', ''))[:500]}"
            for row in rows
        )


@dataclass
class AgentResult:
    agent: str
    status: AgentStatus
    proposed_action: AgentAction
    reason: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def trace_dict(self) -> dict[str, Any]:
        """Keep trace rows compact; full artifacts live in the blackboard table."""
        payload = self.to_dict()
        payload["artifacts"] = [
            {
                "kind": artifact.get("kind", "generic"),
                **{
                    key: value
                    for key, value in artifact.items()
                    if key not in {"kind", "payload"}
                },
                "payload_summary": self._payload_summary(artifact.get("payload")),
            }
            for artifact in self.artifacts
        ]
        return payload

    @staticmethod
    def _payload_summary(payload: Any) -> dict[str, Any]:
        if isinstance(payload, list):
            return {"type": "list", "count": len(payload)}
        if isinstance(payload, dict):
            return {"type": "object", "keys": sorted(payload)[:20]}
        if isinstance(payload, str):
            return {"type": "text", "characters": len(payload)}
        return {"type": type(payload).__name__}


@dataclass
class SupervisorDecision:
    next_agent: str
    reason: str
    terminal: bool = False
    partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchState:
    task_id: str
    goal: str
    academic: bool
    status: str = "running"
    current_agent: str = "supervisor"
    next_agent: str = "planner"
    research_plan: dict[str, Any] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    previous_queries: list[str] = field(default_factory=list)
    search_round: int = 0
    revision_round: int = 0
    completed_research_rounds: dict[str, int] = field(default_factory=dict)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    search_providers: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    critique: dict[str, Any] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    coverage_score: float = 0.0
    citation_score: float = 0.0
    retry_counts: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    failed_agent: str | None = None
    last_result: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchState":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})


ToolCallable = Callable[..., Any] | Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Runtime-enforced tool permissions for independently scoped agents."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolCallable] = {}
        self._grants: dict[str, frozenset[str]] = {}

    def register(self, name: str, tool: ToolCallable) -> None:
        self._tools[name] = tool

    def grant(self, agent: AgentSpec) -> None:
        unknown = agent.allowed_tools.difference(self._tools)
        if unknown:
            raise KeyError(f"Agent {agent.name} references unknown tools: {sorted(unknown)}")
        self._grants[agent.name] = agent.allowed_tools

    def allowed_tools(self, agent_name: str) -> list[str]:
        return sorted(self._grants.get(agent_name, frozenset()))

    async def call(self, agent_name: str, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self._grants.get(agent_name, frozenset()):
            raise ToolPermissionError(
                f"Agent {agent_name} is not allowed to call tool {tool_name}"
            )
        tool = self._tools[tool_name]
        result = tool(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


class AgentJournal:
    """Persists private agent context and shared blackboard artifacts."""

    def __init__(self, db: Any):
        self.db = db

    def context(self, task_id: str, spec: AgentSpec) -> AgentContext:
        messages = self.db.agent_messages(task_id, spec.name, limit=12)
        return AgentContext(task_id=task_id, agent=spec, private_messages=messages)

    def record_message(
        self, task_id: str, agent: str, role: str, content: str
    ) -> None:
        self.db.add_agent_message(task_id, agent, role, content)

    def record_artifacts(
        self, task_id: str, agent: str, artifacts: list[dict[str, Any]]
    ) -> None:
        for artifact in artifacts:
            kind = str(artifact.get("kind", "generic"))
            self.db.add_agent_artifact(task_id, agent, kind, artifact)

    @staticmethod
    def compact_result(result: AgentResult) -> str:
        return json.dumps(
            {
                "status": result.status,
                "proposed_action": result.proposed_action,
                "reason": result.reason,
                "missing_information": result.missing_information,
                "suggested_queries": result.suggested_queries,
                "confidence": result.confidence,
                "metrics": result.metrics,
            },
            ensure_ascii=False,
        )
