from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError

from .settings import Settings


class ProviderUnavailable(RuntimeError):
    pass


async def _retry(call, retries: int):
    for attempt in range(retries + 1):
        try:
            return await call()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
            retryable = (
                isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
                or status in {408, 429, 500, 502, 503, 504}
                or "connection error" in str(exc).lower()
                or "timed out" in str(exc).lower()
            )
            if attempt >= retries or not retryable:
                raise
            await asyncio.sleep(min(0.5 * (2**attempt) + random.random() * 0.25, 8))


class TavilySearch:
    endpoint = "https://api.tavily.com/search"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(self.settings.tavily_api_key)

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        if not self.configured:
            raise ProviderUnavailable("TAVILY_API_KEY 未配置，不能执行实时网络检索")
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results or self.settings.max_search_results,
        }
        async def call():
            async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        data = await _retry(call, self.settings.provider_retries)
        return [
            {
                "title": item.get("title") or item.get("url"),
                "url": item.get("url"),
                "snippet": item.get("content") or "",
                "score": float(item.get("score") or 0),
                "published_at": item.get("published_date"),
            }
            for item in data.get("results", [])
            if item.get("url")
        ]

    async def health(self) -> dict[str, Any]:
        return {"provider": "tavily", "configured": self.configured, "status": "configured_unverified" if self.configured else "missing_key"}


class LLMGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "not-configured")

    @property
    def configured(self) -> bool:
        return bool(self.settings.llm_api_key and self.settings.llm_model)

    async def complete(self, system: str, user: str, *, temperature: float = 0.1, json_mode: bool = False, max_tokens: int = 4096, stream: bool = False) -> str:
        if not self.configured:
            raise ProviderUnavailable("LLM_API_KEY 未配置，不能执行智能体规划、核验和报告生成")
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            kwargs["stream"] = True
        async def call():
            response = await self.client.chat.completions.create(**kwargs)
            if not stream:
                return response.choices[0].message.content or ""
            chunks: list[str] = []
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            return "".join(chunks)
        response = await _retry(call, self.settings.provider_retries)
        return response

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Accept JSON mode responses that still contain a Markdown fence or prose."""
        raw = raw.strip()
        if not raw:
            raise ValueError("empty response")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start : end + 1])
            raise exc

    async def json(self, system: str, user: str) -> dict[str, Any]:
        """Generate structured output with a small retry budget for JSON-mode quirks."""
        last_error: Exception | None = None
        retry_instruction = ""
        for attempt in range(self.settings.json_retries + 1):
            raw = await self.complete(
                system,
                user + retry_instruction,
                json_mode=True,
                max_tokens=2500,
            )
            try:
                parsed = self._parse_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("response is not a JSON object")
                return parsed
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < self.settings.json_retries:
                    retry_instruction = (
                        "\n\n上一次输出无法解析。请只返回一个完整、合法的 JSON 对象，"
                        "不要使用 Markdown 代码块、解释文字或省略号。"
                    )

        detail = "空响应" if isinstance(last_error, ValueError) and str(last_error) == "empty response" else "格式错误"
        raise ProviderUnavailable(
            f"大模型未返回有效 JSON（{detail}），已重试 {self.settings.json_retries} 次。"
            "请检查模型是否支持 JSON 模式，或稍后从断点重试。"
        ) from last_error

    async def health(self) -> dict[str, Any]:
        return {
            "provider": "openai_compatible",
            "configured": self.configured,
            "status": "configured_unverified" if self.configured else "missing_key",
            "base_url": self.settings.llm_base_url,
            "model": self.settings.llm_model,
        }
