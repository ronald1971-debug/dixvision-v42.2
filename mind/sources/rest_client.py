"""
mind/sources/rest_client.py
Minimal REST client using urllib. Caller is responsible for rate limits and
auth. Returns raw text + status so tests don't need a real network layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class HttpResponse:
    status: int
    text: str

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except Exception:
            return None


def get(url: str, headers: dict[str, str] | None = None, timeout_s: float = 5.0) -> HttpResponse:
    req = request.Request(url, headers=headers or {})
    try:
        with request.urlopen(req, timeout=timeout_s) as r:
            body = r.read().decode("utf-8", errors="replace")
            return HttpResponse(status=int(r.status), text=body)
    except error.HTTPError as e:
        return HttpResponse(status=int(e.code), text=e.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return HttpResponse(status=0, text=str(e))
