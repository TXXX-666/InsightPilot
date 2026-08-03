from __future__ import annotations

import json
import re
from typing import Any, Iterable
from urllib.parse import quote_plus

from .mcp_client import McpManager


ACADEMIC_KEYWORDS = (
    "arxiv",
    "paper",
    "papers",
    "literature review",
    "systematic review",
    "survey paper",
    "citation",
    "benchmark",
    "state of the art",
    "sota",
    "论文",
    "文献",
    "学术",
    "综述",
    "科研",
    "引用",
    "基准测试",
    "会议论文",
    "期刊论文",
)

ARXIV_ID = re.compile(
    r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)?"
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def is_academic_goal(goal: str) -> bool:
    normalized = goal.casefold()
    return any(keyword in normalized for keyword in ACADEMIC_KEYWORDS)


def _first(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    lowered = {str(key).casefold(): value for key, value in mapping.items()}
    for name in names:
        value = lowered.get(name.casefold())
        if value not in (None, "", [], {}):
            return value
    return None


def _arxiv_id(value: Any) -> str | None:
    match = ARXIV_ID.search(str(value or ""))
    return match.group("id") if match else None


def _paper_from_mapping(item: dict[str, Any], rank: int) -> dict[str, Any] | None:
    title = _first(item, ("title", "paper_title", "name"))
    abstract = _first(item, ("abstract", "summary", "snippet", "content", "description"))
    identifier_value = _first(
        item,
        ("arxiv_id", "paper_id", "id", "entry_id", "url", "paper_url", "pdf_url", "link"),
    )
    identifier = _arxiv_id(identifier_value)
    url = _first(item, ("url", "paper_url", "entry_id", "link"))
    if identifier:
        url = f"https://arxiv.org/abs/{identifier}"
    if not title or not url:
        return None
    published = _first(item, ("published", "published_at", "publication_date", "date", "updated"))
    authors = _first(item, ("authors", "author"))
    if isinstance(authors, list):
        authors = ", ".join(str(author) for author in authors)
    snippet_parts = [str(part).strip() for part in (authors, abstract) if part]
    score = _first(item, ("score", "relevance_score"))
    try:
        normalized_score = float(score)
    except (TypeError, ValueError):
        normalized_score = max(0.55, 0.96 - rank * 0.03)
    return {
        "url": str(url),
        "title": str(title).strip(),
        "snippet": "\n\n".join(snippet_parts)[:12000],
        "published_at": str(published) if published else None,
        "score": min(max(normalized_score, 0.0), 1.0),
        "provider": "mcp:arxiv",
    }


def _walk_mappings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _json_values(text: str) -> list[Any]:
    values: list[Any] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def normalize_arxiv_response(text: str, query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """Normalize common JSON and Markdown arXiv MCP responses.

    MCP servers are not standardized above the protocol layer: tool names and
    result shapes differ. This parser keeps the product pipeline independent of
    a particular hosted implementation while retaining real arXiv URLs.
    """
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in _json_values(text):
        for mapping in _walk_mappings(value):
            paper = _paper_from_mapping(mapping, len(papers))
            if paper and paper["url"] not in seen:
                seen.add(paper["url"])
                papers.append(paper)
                if len(papers) >= max_results:
                    return papers

    blocks = re.split(r"\n\s*\n|(?=\n?#{1,4}\s)", text)
    for block in blocks:
        identifier = _arxiv_id(block)
        if not identifier:
            continue
        url = f"https://arxiv.org/abs/{identifier}"
        if url in seen:
            continue
        title_match = re.search(r"(?:^|\n)(?:title|论文标题)\s*[:：]\s*(.+)", block, re.IGNORECASE)
        title = title_match.group(1).strip(" #*-") if title_match else f"arXiv:{identifier}"
        abstract_match = re.search(
            r"(?:abstract|summary|摘要)\s*[:：]\s*(.+)", block, re.IGNORECASE | re.DOTALL
        )
        snippet = abstract_match.group(1).strip() if abstract_match else block.strip()
        papers.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet[:12000],
                "published_at": None,
                "score": max(0.55, 0.96 - len(papers) * 0.03),
                "provider": "mcp:arxiv",
            }
        )
        seen.add(url)
        if len(papers) >= max_results:
            break

    if not papers and "arxiv.org" in text.casefold():
        # Keep failure explicit: a search page is traceable but is not presented
        # as an individual paper or as evidence that a specific paper was read.
        return [{
            "url": f"https://arxiv.org/search/?query={quote_plus(query)}&searchtype=all",
            "title": f"arXiv search: {query}",
            "snippet": text[:12000],
            "published_at": None,
            "score": 0.5,
            "provider": "mcp:arxiv",
        }]
    return papers


class ArxivMcpSearch:
    SEARCH_TOOL_NAMES = (
        "search_papers",
        "search_arxiv",
        "arxiv_search",
        "paper_search",
        "search",
    )

    def __init__(self, manager: McpManager):
        self.manager = manager

    def _search_tool(self) -> dict[str, Any] | None:
        tools = self.manager.tools_for_server("arxiv")
        by_name = {str(tool.get("name", "")).casefold(): tool for tool in tools}
        for candidate in self.SEARCH_TOOL_NAMES:
            if candidate in by_name:
                return by_name[candidate]
        for tool in tools:
            searchable = f"{tool.get('name', '')} {tool.get('description', '')}".casefold()
            if "search" in searchable and ("paper" in searchable or "arxiv" in searchable):
                return tool
        return None

    @property
    def configured(self) -> bool:
        return self._search_tool() is not None

    @staticmethod
    def _arguments(tool: dict[str, Any], query: str, max_results: int) -> dict[str, Any]:
        schema = tool.get("inputSchema") or {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        args: dict[str, Any] = {}

        query_fields = ("query", "search_query", "query_string", "keywords", "term")
        query_field = next((name for name in query_fields if name in properties), None)
        if query_field is None and required:
            query_field = str(required[0])
        if query_field is None:
            query_field = "query"
        args[query_field] = query

        for limit_field in ("max_results", "limit", "maxResults", "count", "page_size"):
            if limit_field in properties:
                args[limit_field] = max_results
                break
        return args

    async def search(self, query: str, max_results: int = 8) -> list[dict[str, Any]]:
        tool = self._search_tool()
        if tool is None:
            raise RuntimeError("已连接的 arXiv MCP 未发现论文搜索工具")
        args = self._arguments(tool, query, max_results)
        response = await self.manager.call_tool(str(tool["prefixedName"]), args)
        return normalize_arxiv_response(response, query, max_results)
