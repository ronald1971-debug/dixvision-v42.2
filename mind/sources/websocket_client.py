"""
mind/sources/websocket_client.py
Thin abstraction over a websocket client. The concrete websocket lib is
pluggable (``websockets``, ``websocket-client``, etc.). Default impl is a
stub that only records registrations — connection code is injected by users.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

OnMessage = Callable[[str, str], None]  # (channel, message)


@dataclass
class Subscription:
    url: str
    channel: str
    on_message: OnMessage


@dataclass
class WebSocketClient:
    subs: dict[str, Subscription] = field(default_factory=dict)

    def subscribe(self, name: str, url: str, channel: str, on_message: OnMessage) -> None:
        self.subs[name] = Subscription(url=url, channel=channel, on_message=on_message)

    def unsubscribe(self, name: str) -> None:
        self.subs.pop(name, None)

    def active(self) -> dict[str, Subscription]:
        return dict(self.subs)


_client: WebSocketClient | None = None


def get_websocket_client() -> WebSocketClient:
    global _client
    if _client is None:
        _client = WebSocketClient()
    return _client
