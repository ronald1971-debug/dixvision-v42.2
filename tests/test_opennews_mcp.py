"""Tests for the OpenNews MCP server (D4)."""

from __future__ import annotations

import io
import json

from intelligence_engine.knowledge import NewsKnowledgeIndex
from intelligence_engine.mcp import (
    OPENNEWS_PROTOCOL_VERSION,
    OPENNEWS_SERVER_NAME,
    OpenNewsServer,
)


def _req(method: str, *, params: object | None = None, rid: int = 1) -> dict:
    out: dict = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        out["params"] = params
    return out


def test_initialize_returns_protocol_version_and_name() -> None:
    server = OpenNewsServer()
    resp = server.handle_request(_req("initialize"))
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == OPENNEWS_PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == OPENNEWS_SERVER_NAME


def test_tools_list_returns_five_tools() -> None:
    server = OpenNewsServer()
    resp = server.handle_request(_req("tools/list"))
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {
        "ingest_news",
        "query_similar",
        "index_stats",
        "list_sources",
        "drop_news",
    }


def test_ingest_news_then_query_similar() -> None:
    server = OpenNewsServer()
    ingest = server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "ingest_news",
                "arguments": {
                    "ts_ns": 1,
                    "source": "COINDESK",
                    "guid": "a",
                    "title": "Bitcoin ETF inflows surge",
                },
            },
        )
    )
    assert ingest["result"]["added"] is True
    server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "ingest_news",
                "arguments": {
                    "ts_ns": 2,
                    "source": "COINDESK",
                    "guid": "b",
                    "title": "Cloudy weather report",
                },
            },
        )
    )
    query = server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "query_similar",
                "arguments": {"text": "bitcoin etf", "top_k": 1},
            },
        )
    )
    hits = query["result"]["hits"]
    assert len(hits) == 1
    assert hits[0]["item"]["guid"] == "a"
    assert hits[0]["score"] > 0.0


def test_ingest_news_duplicate_returns_added_false() -> None:
    server = OpenNewsServer()
    args = {
        "name": "ingest_news",
        "arguments": {
            "ts_ns": 1,
            "source": "COINDESK",
            "guid": "x",
            "title": "hello",
        },
    }
    a = server.handle_request(_req("tools/call", params=args))
    b = server.handle_request(_req("tools/call", params=args))
    assert a["result"]["added"] is True
    assert b["result"]["added"] is False


def test_drop_news_removes_row() -> None:
    server = OpenNewsServer()
    server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "ingest_news",
                "arguments": {
                    "ts_ns": 1,
                    "source": "COINDESK",
                    "guid": "x",
                    "title": "hello",
                },
            },
        )
    )
    a = server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "drop_news",
                "arguments": {"source": "COINDESK", "guid": "x"},
            },
        )
    )
    b = server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "drop_news",
                "arguments": {"source": "COINDESK", "guid": "x"},
            },
        )
    )
    assert a["result"]["dropped"] is True
    assert b["result"]["dropped"] is False


def test_index_stats_and_list_sources() -> None:
    idx = NewsKnowledgeIndex(max_items=8)
    server = OpenNewsServer(idx)
    for source in ("COINDESK", "REUTERS", "COINDESK"):
        server.handle_request(
            _req(
                "tools/call",
                params={
                    "name": "ingest_news",
                    "arguments": {
                        "ts_ns": 1 + len(idx),
                        "source": source,
                        "guid": f"g{len(idx)}",
                        "title": "hello world",
                    },
                },
            )
        )
    stats = server.handle_request(
        _req("tools/call", params={"name": "index_stats", "arguments": {}})
    )["result"]
    assert stats["size"] == 3
    assert stats["max_items"] == 8
    assert stats["unique_sources"] == 2
    sources = server.handle_request(
        _req(
            "tools/call",
            params={"name": "list_sources", "arguments": {}},
        )
    )["result"]["sources"]
    assert sources == ["COINDESK", "REUTERS"]


def test_unknown_method_returns_method_not_found() -> None:
    server = OpenNewsServer()
    resp = server.handle_request(_req("nope"))
    assert resp["error"]["code"] == -32601


def test_unknown_tool_returns_method_not_found() -> None:
    server = OpenNewsServer()
    resp = server.handle_request(
        _req(
            "tools/call",
            params={"name": "bogus", "arguments": {}},
        )
    )
    assert resp["error"]["code"] == -32601


def test_invalid_jsonrpc_version_rejected() -> None:
    server = OpenNewsServer()
    resp = server.handle_request(
        {"jsonrpc": "1.0", "id": 1, "method": "initialize"}
    )
    assert resp["error"]["code"] == -32600


def test_invalid_params_for_query_similar() -> None:
    server = OpenNewsServer()
    resp = server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "query_similar",
                "arguments": {"text": 123},
            },
        )
    )
    assert resp["error"]["code"] == -32602


def test_invalid_top_k_returns_invalid_params() -> None:
    server = OpenNewsServer()
    resp = server.handle_request(
        _req(
            "tools/call",
            params={
                "name": "query_similar",
                "arguments": {"text": "hello", "top_k": 0},
            },
        )
    )
    assert resp["error"]["code"] == -32602


def test_serve_stdio_handles_one_request_then_eof() -> None:
    server = OpenNewsServer()
    stdin = io.BytesIO(
        (json.dumps(_req("initialize")) + "\n").encode("utf-8")
    )
    stdout = io.BytesIO()
    server.serve_stdio(stdin, stdout)
    line = stdout.getvalue().decode("utf-8").strip()
    resp = json.loads(line)
    assert resp["result"]["serverInfo"]["name"] == OPENNEWS_SERVER_NAME


def test_serve_stdio_handles_parse_error() -> None:
    server = OpenNewsServer()
    stdin = io.BytesIO(b"{not json\n")
    stdout = io.BytesIO()
    server.serve_stdio(stdin, stdout)
    resp = json.loads(stdout.getvalue().decode("utf-8").strip())
    assert resp["error"]["code"] == -32700
    assert resp["id"] is None
