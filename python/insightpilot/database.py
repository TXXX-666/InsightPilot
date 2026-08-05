from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL DEFAULT 'default',
  goal TEXT NOT NULL,
  status TEXT NOT NULL,
  mode TEXT NOT NULL DEFAULT 'research',
  error TEXT,
  result_json TEXT,
  report_path TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  agent TEXT,
  message TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  url TEXT NOT NULL,
  title TEXT,
  published_at TEXT,
  retrieved_at TEXT NOT NULL,
  source_type TEXT NOT NULL,
  fetch_status TEXT NOT NULL DEFAULT 'unknown',
  content_hash TEXT NOT NULL,
  raw_text TEXT,
  UNIQUE(task_id, url),
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  claim TEXT,
  quote TEXT NOT NULL,
  reliability REAL NOT NULL DEFAULT 0.5,
  stance TEXT NOT NULL DEFAULT 'supports',
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(task_id, content_hash),
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
  FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claims (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  text TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  status TEXT NOT NULL DEFAULT 'unverified',
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_evidence (
  claim_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  relation TEXT NOT NULL DEFAULT 'supports',
  PRIMARY KEY(claim_id, evidence_id),
  FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE,
  FOREIGN KEY(evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_verifications (
  claim_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  entailment_score REAL NOT NULL DEFAULT 0,
  supporting_quote TEXT,
  reason TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY(claim_id, evidence_id),
  FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE,
  FOREIGN KEY(evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS verification_batches (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  cache_key TEXT NOT NULL,
  evidence_ids_json TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT,
  error TEXT,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  applied INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(task_id, cache_key),
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_checkpoints (
  task_id TEXT PRIMARY KEY,
  current_agent TEXT NOT NULL,
  state_version INTEGER NOT NULL DEFAULT 1,
  state_json TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_runs (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  objective TEXT NOT NULL,
  allowed_tools_json TEXT NOT NULL,
  round INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  input_json TEXT,
  output_json TEXT,
  error TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_artifacts (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  kind TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(task_id, agent, kind, content_hash),
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS monitors (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  query TEXT NOT NULL,
  interval_minutes INTEGER NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  notify_channel TEXT,
  last_fingerprint TEXT,
  last_run_at TEXT,
  next_run_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitor_runs (
  id TEXT PRIMARY KEY,
  monitor_id TEXT NOT NULL,
  task_id TEXT,
  status TEXT NOT NULL,
  changed INTEGER NOT NULL DEFAULT 0,
  fingerprint TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  FOREIGN KEY(monitor_id) REFERENCES monitors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY,
  action_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  decided_at TEXT,
  decision_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_id, id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_monitors_due ON monitors(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_agent_runs_task ON agent_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_agent_messages_scope ON agent_messages(task_id, agent, id);
CREATE INDEX IF NOT EXISTS idx_agent_artifacts_task ON agent_artifacts(task_id, kind);
CREATE INDEX IF NOT EXISTS idx_verification_batches_task
  ON verification_batches(task_id, model, prompt_version, status, applied);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            task_columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
            if "attempt_count" not in task_columns:
                conn.execute("ALTER TABLE tasks ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0")
            source_columns = {row[1] for row in conn.execute("PRAGMA table_info(sources)")}
            if "fetch_status" not in source_columns:
                conn.execute("ALTER TABLE sources ADD COLUMN fetch_status TEXT NOT NULL DEFAULT 'unknown'")
            conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(sql, params)
            conn.commit()

    def insert(self, table: str, data: dict[str, Any]) -> None:
        columns = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        self.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(data.values()))

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def create_task(self, goal: str, project_id: str = "default", mode: str = "research") -> str:
        task_id = str(uuid.uuid4())
        now = utcnow()
        self.insert("tasks", {"id": task_id, "project_id": project_id, "goal": goal, "status": "queued", "mode": mode, "created_at": now, "updated_at": now})
        self.add_event(task_id, "task.queued", "supervisor", "任务已进入后台队列")
        return task_id

    def add_event(self, task_id: str, event_type: str, agent: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.insert("events", {"task_id": task_id, "event_type": event_type, "agent": agent, "message": message, "payload_json": json.dumps(payload or {}, ensure_ascii=False), "created_at": utcnow()})

    def update_task(self, task_id: str, status: str, **fields: Any) -> None:
        fields.update(status=status, updated_at=utcnow())
        if status == "running":
            fields.setdefault("started_at", utcnow())
        if status in {"completed", "failed", "cancelled", "partial"}:
            fields.setdefault("finished_at", utcnow())
        assignments = ", ".join(f"{key}=?" for key in fields)
        self.execute(f"UPDATE tasks SET {assignments} WHERE id=?", tuple(fields.values()) + (task_id,))

    def task(self, task_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM tasks WHERE id=?", (task_id,))
        if row and row.get("result_json"):
            row["result"] = json.loads(row["result_json"])
        return row

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.fetchall("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))

    def save_verification_batch(
        self,
        task_id: str,
        cache_key: str,
        evidence_ids: list[str],
        status: str,
        model: str,
        prompt_version: str,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = utcnow()
        self.execute(
            """
            INSERT INTO verification_batches(
              id, task_id, cache_key, evidence_ids_json, status, payload_json,
              error, model, prompt_version, applied, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(task_id, cache_key) DO UPDATE SET
              evidence_ids_json=excluded.evidence_ids_json,
              status=excluded.status,
              payload_json=excluded.payload_json,
              error=excluded.error,
              model=excluded.model,
              prompt_version=excluded.prompt_version,
              applied=CASE WHEN excluded.status='completed' THEN 0 ELSE verification_batches.applied END,
              updated_at=excluded.updated_at
            """,
            (
                str(uuid.uuid4()),
                task_id,
                cache_key,
                json.dumps(sorted(set(evidence_ids)), ensure_ascii=False),
                status,
                json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                error,
                model,
                prompt_version,
                now,
                now,
            ),
        )

    def completed_verification_evidence(
        self, task_id: str, model: str, prompt_version: str
    ) -> set[str]:
        rows = self.fetchall(
            "SELECT evidence_ids_json FROM verification_batches "
            "WHERE task_id=? AND model=? AND prompt_version=? AND status='completed'",
            (task_id, model, prompt_version),
        )
        return {
            str(evidence_id)
            for row in rows
            for evidence_id in json.loads(row["evidence_ids_json"] or "[]")
        }

    def unapplied_verification_batches(
        self, task_id: str, model: str, prompt_version: str
    ) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT cache_key, evidence_ids_json, payload_json FROM verification_batches "
            "WHERE task_id=? AND model=? AND prompt_version=? "
            "AND status='completed' AND applied=0 ORDER BY created_at, id",
            (task_id, model, prompt_version),
        )
        batches: list[dict[str, Any]] = []
        for row in rows:
            if not row.get("payload_json"):
                continue
            batches.append(
                {
                    "cache_key": row["cache_key"],
                    "evidence_ids": json.loads(row["evidence_ids_json"] or "[]"),
                    "payload": json.loads(row["payload_json"]),
                }
            )
        return batches

    def replace_verified_claims(
        self,
        task_id: str,
        claims: list[dict[str, Any]],
        applied_batch_keys: list[str],
    ) -> None:
        """Replace claims and mark their source batches applied in one transaction."""
        relation_priority = {
            "contradicted": 4,
            "partial": 3,
            "entailed": 2,
            "irrelevant": 1,
        }
        with self._lock, self.connect() as conn:
            conn.execute("DELETE FROM claims WHERE task_id=?", (task_id,))
            for claim in claims:
                claim_id = str(claim["claim_id"])
                conn.execute(
                    "INSERT INTO claims(id, task_id, text, confidence, status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        claim_id,
                        task_id,
                        claim["text"],
                        claim["confidence"],
                        claim["status"],
                        claim.get("created_at") or utcnow(),
                    ),
                )
                links_by_evidence: dict[str, dict[str, Any]] = {}
                for link in claim.get("verifications", []):
                    evidence_id = str(link.get("evidence_id", ""))
                    relation = str(link.get("relation", "irrelevant"))
                    if not evidence_id or relation == "irrelevant":
                        continue
                    candidate = dict(link, evidence_id=evidence_id, relation=relation)
                    current = links_by_evidence.get(evidence_id)
                    candidate_rank = (
                        relation_priority.get(relation, 0),
                        float(candidate.get("entailment_score", 0) or 0),
                    )
                    current_rank = (
                        relation_priority.get(str(current.get("relation")), 0),
                        float(current.get("entailment_score", 0) or 0),
                    ) if current else (-1, -1.0)
                    if candidate_rank > current_rank:
                        links_by_evidence[evidence_id] = candidate

                for link in links_by_evidence.values():
                    conn.execute(
                        "INSERT INTO claim_verifications("
                        "claim_id, evidence_id, relation, entailment_score, "
                        "supporting_quote, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            claim_id,
                            link["evidence_id"],
                            link["relation"],
                            float(link.get("entailment_score", 0) or 0),
                            str(link.get("supporting_quote", "")),
                            str(link.get("reason", ""))[:1000],
                            utcnow(),
                        ),
                    )
                    if link["relation"] in {"entailed", "partial"}:
                        conn.execute(
                            "INSERT INTO claim_evidence(claim_id, evidence_id, relation) "
                            "VALUES (?, ?, ?)",
                            (
                                claim_id,
                                link["evidence_id"],
                                "supports" if link["relation"] == "entailed" else "partial",
                            ),
                        )

            conn.executemany(
                "UPDATE verification_batches SET applied=1, updated_at=? "
                "WHERE task_id=? AND cache_key=? AND status='completed'",
                [(utcnow(), task_id, key) for key in dict.fromkeys(applied_batch_keys)],
            )
            conn.commit()

    def save_checkpoint(
        self, task_id: str, current_agent: str, state: dict[str, Any]
    ) -> None:
        payload = json.dumps(state, ensure_ascii=False)
        self.execute(
            """
            INSERT INTO workflow_checkpoints(task_id, current_agent, state_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
              current_agent=excluded.current_agent,
              state_version=workflow_checkpoints.state_version+1,
              state_json=excluded.state_json,
              updated_at=excluded.updated_at
            """,
            (task_id, current_agent, payload, utcnow()),
        )

    def checkpoint(self, task_id: str) -> dict[str, Any] | None:
        row = self.fetchone(
            "SELECT * FROM workflow_checkpoints WHERE task_id=?", (task_id,)
        )
        if not row:
            return None
        row["state"] = json.loads(row["state_json"])
        return row

    def start_agent_run(
        self,
        task_id: str,
        agent: str,
        objective: str,
        allowed_tools: list[str],
        round_number: int,
        input_payload: dict[str, Any],
    ) -> str:
        run_id = str(uuid.uuid4())
        self.insert(
            "agent_runs",
            {
                "id": run_id,
                "task_id": task_id,
                "agent": agent,
                "objective": objective,
                "allowed_tools_json": json.dumps(allowed_tools, ensure_ascii=False),
                "round": round_number,
                "status": "running",
                "input_json": json.dumps(input_payload, ensure_ascii=False),
                "started_at": utcnow(),
            },
        )
        return run_id

    def finish_agent_run(
        self,
        run_id: str,
        status: str,
        output_payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.execute(
            "UPDATE agent_runs SET status=?, output_json=?, error=?, finished_at=? WHERE id=?",
            (
                status,
                json.dumps(output_payload or {}, ensure_ascii=False),
                error,
                utcnow(),
                run_id,
            ),
        )

    def add_agent_message(
        self, task_id: str, agent: str, role: str, content: str
    ) -> None:
        self.insert(
            "agent_messages",
            {
                "task_id": task_id,
                "agent": agent,
                "role": role,
                "content": content,
                "created_at": utcnow(),
            },
        )

    def agent_messages(
        self, task_id: str, agent: str, limit: int = 12
    ) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT role, content, created_at FROM agent_messages "
            "WHERE task_id=? AND agent=? ORDER BY id DESC LIMIT ?",
            (task_id, agent, limit),
        )
        rows.reverse()
        return rows

    def add_agent_artifact(
        self, task_id: str, agent: str, kind: str, payload: dict[str, Any]
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.execute(
            "INSERT OR IGNORE INTO agent_artifacts "
            "(id, task_id, agent, kind, content_hash, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), task_id, agent, kind, digest, encoded, utcnow()),
        )

    def agent_trace(self, task_id: str) -> dict[str, Any]:
        runs = self.fetchall(
            "SELECT * FROM agent_runs WHERE task_id=? ORDER BY started_at, id",
            (task_id,),
        )
        for run in runs:
            run["allowed_tools"] = json.loads(run.pop("allowed_tools_json") or "[]")
            run["input"] = json.loads(run.pop("input_json") or "{}")
            run["output"] = json.loads(run.pop("output_json") or "{}")
        artifacts = self.fetchall(
            "SELECT id, agent, kind, payload_json, created_at FROM agent_artifacts "
            "WHERE task_id=? ORDER BY created_at, id",
            (task_id,),
        )
        for artifact in artifacts:
            payload = json.loads(artifact.pop("payload_json"))
            content = payload.get("payload") if isinstance(payload, dict) else payload
            if isinstance(content, list):
                artifact["payload_summary"] = {"type": "list", "count": len(content)}
            elif isinstance(content, dict):
                artifact["payload_summary"] = {"type": "object", "keys": sorted(content)[:20]}
            elif isinstance(content, str):
                artifact["payload_summary"] = {"type": "text", "characters": len(content)}
            else:
                artifact["payload_summary"] = {"type": type(content).__name__}
        checkpoint = self.checkpoint(task_id)
        if checkpoint:
            checkpoint.pop("state_json", None)
            state = checkpoint.pop("state")
            checkpoint["state"] = {
                "status": state.get("status"),
                "current_agent": state.get("current_agent"),
                "next_agent": state.get("next_agent"),
                "search_round": state.get("search_round", 0),
                "revision_round": state.get("revision_round", 0),
                "coverage_score": state.get("coverage_score", 0),
                "citation_score": state.get("citation_score", 0),
                "partial": state.get("partial", False),
                "failed_agent": state.get("failed_agent"),
                "evidence_count": len(state.get("evidence", [])),
                "claim_count": len(state.get("claims", [])),
                "error_count": len(state.get("errors", [])),
                "last_error": (state.get("errors") or [None])[-1],
            }
        return {"runs": runs, "artifacts": artifacts, "checkpoint": checkpoint}
