from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    db_path: Path = Path(os.getenv("INSIGHTPILOT_DB_PATH", str(PROJECT_ROOT / "data" / "insightpilot.db")))
    report_dir: Path = PROJECT_ROOT / "reports"
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    feishu_webhook_url: str = os.getenv("FEISHU_WEBHOOK_URL", "")
    api_url: str = os.getenv("INSIGHTPILOT_API_URL", "http://127.0.0.1:8000")
    host: str = os.getenv("INSIGHTPILOT_HOST", "127.0.0.1")
    port: int = _int("INSIGHTPILOT_PORT", 8000)
    workers: int = _int("INSIGHTPILOT_WORKERS", 2)
    max_search_results: int = _int("INSIGHTPILOT_MAX_SEARCH_RESULTS", 8)
    fetch_concurrency: int = _int("INSIGHTPILOT_FETCH_CONCURRENCY", 5)
    provider_retries: int = _int("INSIGHTPILOT_PROVIDER_RETRIES", 3)
    json_retries: int = _int("INSIGHTPILOT_JSON_RETRIES", 2)
    report_max_tokens: int = _int("INSIGHTPILOT_REPORT_MAX_TOKENS", 8192)

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
