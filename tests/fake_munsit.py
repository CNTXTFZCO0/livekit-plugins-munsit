# Copyright 2026 LiveKit, Inc.
"""Local WebSocket server emulating Munsit's STT streaming endpoint for tests."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web


@dataclass
class FakeMunsitState:
    base_url: str = ""
    received_messages: list[dict[str, Any]] = field(default_factory=list)
    received_headers: dict[str, str] = field(default_factory=dict)
    received_query: dict[str, str] = field(default_factory=dict)
    # Each script entry is (delay_seconds, event_type, payload).
    script: list[tuple[float, str, str]] = field(default_factory=list)
    require_api_key: str | None = None
    close_after_messages: int | None = None  # forces server-side WS close after N audio chunks


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    state: FakeMunsitState = request.app["state"]
    state.received_headers = dict(request.headers)
    state.received_query = dict(request.query)

    # Optional auth check
    if state.require_api_key is not None:
        provided = (
            request.headers.get("x-api-key")
            or _strip_bearer(request.headers.get("Authorization"))
            or request.query.get("token")
        )
        if provided != state.require_api_key:
            return web.Response(status=401, text="invalid api key")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async def _replay_script() -> None:
        for delay, event, payload in state.script:
            await asyncio.sleep(delay)
            if ws.closed:
                return
            await ws.send_str(json.dumps({"event": event, "data": payload}))

    replay_task = asyncio.create_task(_replay_script())
    chunks_seen = 0

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    parsed = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                state.received_messages.append(parsed)
                chunks_seen += 1
                if (
                    state.close_after_messages is not None
                    and chunks_seen >= state.close_after_messages
                ):
                    await ws.close()
                    break
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        replay_task.cancel()
        try:
            await replay_task
        except asyncio.CancelledError:
            pass

    return ws


def _strip_bearer(header_value: str | None) -> str | None:
    if header_value and header_value.lower().startswith("bearer "):
        return header_value[7:].strip()
    return None


def make_app(state: FakeMunsitState) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_get("/api/v1/websocket/speech-to-text", _ws_handler)
    return app


class FakeMunsitServer:
    """Convenience wrapper. Use via the ``fake_munsit`` pytest fixture."""

    def __init__(self) -> None:
        self.state = FakeMunsitState()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.url: str = ""

    async def start(self) -> str:
        app = make_app(self.state)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host="127.0.0.1", port=0)
        await self._site.start()
        # Read the bound port from the site's server.
        assert self._site._server and self._site._server.sockets  # type: ignore[attr-defined]
        port = self._site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
        self.url = f"ws://127.0.0.1:{port}/api/v1/websocket/speech-to-text"
        self.state.base_url = self.url
        return self.url

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    def script(self, entries: list[tuple[float, str, str]]) -> None:
        self.state.script = list(entries)
