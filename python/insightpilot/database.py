from __future__ import annotations

import json
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
