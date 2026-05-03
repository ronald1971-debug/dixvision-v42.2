"""OpenNews MCP server — deterministic JSON-RPC over stdio (D4).

Exposes the :class:`NewsKnowledgeIndex` to MCP clients. The protocol
is a deliberately small subset of MCP (initialize / tools/list /
tools/call) so the server has zero third-party deps; production
clients that already speak the full MCP spec interoperate because
this subset is a strict prefix.

Tools exposed:

* ``ingest_news`` — append one ``NewsItem`` to the index.
* ``query_similar`` — top-k cosine similarity search.
* ``index_stats`` — return :class:`IndexStats`.
* ``list_sources`` — return distinct ``NewsItem.source`` values
  currently in the index.
* ``drop_news`` — drop one ``(source, guid)`` row.

Determinism: the server itself does no I/O beyond stdio framing, no
clock reads, no PRNG. Replays of the same JSON-RPC request stream
produce byte-identical responses (modulo the request ``id`` echoed
back, which is also caller-supplied).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO

from core.contracts.news import NewsItem
from intelligence_engine.knowledge import (
    KNOWLEDGE_INDEX_VERSION,
    NewsKnowledgeIndex,
)

#: Protocol version reported in ``initialize``. Bumped on schema
#: changes; clients should refuse if they don't recognize it.
OPENNEWS_PROTOCOL_VERSION = "1.0"

#: Server name returned in ``initialize``.
OPENNEWS_SERVER_NAME = "opennews-mcp"

# JSON-RPC 2.0 standard error codes.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


@dataclass(frozen=True, slots=True)
class _Tool:
    name: str
    description: str
    input_schema: Mapping[str, Any]


_TOOLS: tuple[_Tool, ...] = (
    _Tool(
        name="ingest_news",
        description=(
            "Append one news item to the deterministic index. Returns "
            "{added: bool} — false when (source, guid) is already known."
        ),
        input_schema={
            "type": "object",
            "required": [
                "ts_ns",
                "source",
                "guid",
                "title",
            ],
            "properties": {
                "ts_ns": {"type": "integer"},
                "source": {"type": "string"},
                "guid": {"type": "string"},
                "title": {"type": "string"},
                "url": {"type": "string"},
                "summary": {"type": "string"},
                "published_ts_ns": {"type": ["integer", "null"]},
                "meta": {"type": "object"},
            },
        },
    ),
    _Tool(
        name="query_similar",
        description=(
            "Top-k cosine similarity search over indexed news. Returns "
            "{hits: [{score, item}]} ordered by descending score."
        ),
        input_schema={
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "min_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                },
                "source": {"type": ["string", "null"]},
            },
        },
    ),
    _Tool(
        name="index_stats",
        description=(
            "Snapshot of the index: size, max_items, version, "
            "unique_tokens, unique_sources."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    _Tool(
        name="list_sources",
        description="Return distinct NewsItem.source values currently indexed.",
        input_schema={"type": "object", "properties": {}},
    ),
    _Tool(
        name="drop_news",
        description=(
            "Drop one (source, guid) row. Returns {dropped: bool}."
        ),
        input_schema={
            "type": "object",
            "required": ["source", "guid"],
            "properties": {
                "source": {"type": "string"},
                "guid": {"type": "string"},
            },
        },
    ),
)


def _news_item_to_dict(item: NewsItem) -> dict[str, Any]:
    return {
        "ts_ns": item.ts_ns,
        "source": item.source,
        "guid": item.guid,
        "title": item.title,
        "url": item.url,
        "summary": item.summary,
        "published_ts_ns": item.published_ts_ns,
        "meta": dict(item.meta),
    }


def _news_item_from_params(params: Mapping[str, Any]) -> NewsItem:
    raw_ts = params.get("ts_ns")
    if not isinstance(raw_ts, int):
        raise ValueError("ts_ns must be int")
    return NewsItem(
        ts_ns=raw_ts,
        source=str(params.get("source", "")),
        guid=str(params.get("guid", "")),
        title=str(params.get("title", "")),
        url=str(params.get("url", "")),
        summary=str(params.get("summary", "")),
        published_ts_ns=(
            int(params["published_ts_ns"])
            if params.get("published_ts_ns") is not None
            else None
        ),
        meta={
            str(k): str(v)
            for k, v in (params.get("meta") or {}).items()
        },
    )


class OpenNewsServer:
    """MCP-style JSON-RPC server wrapping a :class:`NewsKnowledgeIndex`.

    The server is intentionally small — five tools, no streaming, no
    auth, no notifications. It is meant to be embedded by other engines
    (in-process via :meth:`handle_request`) or exposed to external MCP
    clients via :meth:`serve_stdio` (line-delimited JSON-RPC).

    Thread safety: not safe — wrap in a per-process actor / queue.
    """

    def __init__(
        self,
        index: NewsKnowledgeIndex | None = None,
    ) -> None:
        self._index = index if index is not None else NewsKnowledgeIndex()

    @property
    def index(self) -> NewsKnowledgeIndex:
        return self._index

    # -- public dispatch ---------------------------------------------------

    def handle_request(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Dispatch one JSON-RPC request, return one JSON-RPC response."""

        if not isinstance(request, Mapping):
            return _err(None, _INVALID_REQUEST, "request must be object")
        rid = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return _err(rid, _INVALID_REQUEST, "jsonrpc must be '2.0'")
        method = request.get("method")
        if not isinstance(method, str):
            return _err(rid, _INVALID_REQUEST, "method must be string")
        params = request.get("params") or {}
        if not isinstance(params, Mapping):
            return _err(rid, _INVALID_PARAMS, "params must be object")
        try:
            result = self._dispatch(method, params)
        except _ToolError as exc:
            return _err(rid, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001
            return _err(
                rid,
                _INTERNAL_ERROR,
                f"unhandled: {type(exc).__name__}: {exc!s}",
            )
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def serve_stdio(
        self,
        stdin: BinaryIO,
        stdout: BinaryIO,
    ) -> None:
        """Run the JSON-RPC loop until ``stdin`` reaches EOF.

        Each line is one JSON-RPC request; each response is written as
        one line. Malformed JSON yields a parse-error response.
        """

        for raw in stdin:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                resp = _err(None, _PARSE_ERROR, f"json: {exc!s}")
            else:
                resp = self.handle_request(req)
            stdout.write((json.dumps(resp) + "\n").encode("utf-8"))
            stdout.flush()

    # -- internal ----------------------------------------------------------

    def _dispatch(
        self, method: str, params: Mapping[str, Any]
    ) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": OPENNEWS_PROTOCOL_VERSION,
                "serverInfo": {
                    "name": OPENNEWS_SERVER_NAME,
                    "version": KNOWLEDGE_INDEX_VERSION,
                },
                "capabilities": {"tools": {}},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": dict(t.input_schema),
                    }
                    for t in _TOOLS
                ]
            }
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str):
                raise _ToolError(
                    _INVALID_PARAMS, "tools/call.name must be string"
                )
            if not isinstance(arguments, Mapping):
                raise _ToolError(
                    _INVALID_PARAMS,
                    "tools/call.arguments must be object",
                )
            return self._call_tool(name, arguments)
        raise _ToolError(_METHOD_NOT_FOUND, f"unknown method: {method!r}")

    def _call_tool(
        self, name: str, args: Mapping[str, Any]
    ) -> Any:
        if name == "ingest_news":
            try:
                item = _news_item_from_params(args)
            except (ValueError, TypeError) as exc:
                raise _ToolError(_INVALID_PARAMS, str(exc)) from exc
            try:
                added = self._index.add(item)
            except ValueError as exc:
                raise _ToolError(_INVALID_PARAMS, str(exc)) from exc
            return {"added": added}
        if name == "query_similar":
            text = args.get("text")
            if not isinstance(text, str):
                raise _ToolError(
                    _INVALID_PARAMS, "query_similar.text must be string"
                )
            top_k = int(args.get("top_k", 5))
            min_score = float(args.get("min_score", 0.0))
            source_arg = args.get("source")
            source = (
                str(source_arg) if source_arg is not None else None
            )
            try:
                hits = self._index.query(
                    text,
                    top_k=top_k,
                    min_score=min_score,
                    source=source,
                )
            except ValueError as exc:
                raise _ToolError(_INVALID_PARAMS, str(exc)) from exc
            return {
                "hits": [
                    {
                        "score": h.score,
                        "item": _news_item_to_dict(h.item),
                    }
                    for h in hits
                ]
            }
        if name == "index_stats":
            s = self._index.stats()
            return {
                "size": s.size,
                "max_items": s.max_items,
                "version": s.version,
                "unique_tokens": s.unique_tokens,
                "unique_sources": s.unique_sources,
            }
        if name == "list_sources":
            return {"sources": list(self._index.sources())}
        if name == "drop_news":
            source = str(args.get("source", ""))
            guid = str(args.get("guid", ""))
            return {"dropped": self._index.drop(source, guid)}
        raise _ToolError(
            _METHOD_NOT_FOUND, f"unknown tool: {name!r}"
        )


class _ToolError(Exception):
    __slots__ = ("code", "message")

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _err(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": code, "message": message},
    }


__all__ = [
    "OPENNEWS_PROTOCOL_VERSION",
    "OPENNEWS_SERVER_NAME",
    "OpenNewsServer",
]
