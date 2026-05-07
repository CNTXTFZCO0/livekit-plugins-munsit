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

    # ---- Batch endpoint state ----
    batch_received_count: int = 0
    batch_received_headers: dict[str, str] = field(default_factory=dict)
    batch_received_query: dict[str, str] = field(default_factory=dict)
    batch_received_models: list[str] = field(default_factory=list)
    batch_received_audio_sizes: list[int] = field(default_factory=list)
    # Configurable response. Defaults emit a successful transcription with empty timestamps.
    batch_response_status: int = 200
    batch_response_body: dict[str, Any] | None = None  # raw JSON body to return


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


async def _batch_handler(request: web.Request) -> web.Response:
    state: FakeMunsitState = request.app["state"]
    state.batch_received_count += 1
    state.batch_received_headers = dict(request.headers)
    state.batch_received_query = dict(request.query)

    if state.require_api_key is not None:
        provided = (
            request.headers.get("x-api-key")
            or _strip_bearer(request.headers.get("Authorization"))
            or request.query.get("token")
        )
        if provided != state.require_api_key:
            return web.json_response({"statusCode": 401, "message": "Unauthorized"}, status=401)

    reader = await request.multipart()
    seen_model = ""
    seen_audio_size = 0
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            payload = await part.read(decode=False)
            seen_audio_size = len(payload)
        elif part.name == "model":
            seen_model = (await part.read(decode=False)).decode("utf-8")
    state.batch_received_models.append(seen_model)
    state.batch_received_audio_sizes.append(seen_audio_size)

    if state.batch_response_status != 200:
        return web.json_response(
            state.batch_response_body or {"message": "error"},
            status=state.batch_response_status,
        )

    body = state.batch_response_body or {
        "statusCode": 200,
        "data": {
            "transcriptionId": f"fake-{state.batch_received_count}",
            "transcription": "أبي أحول درهم لفرحان",
            "duration": 2.6,
            "timestamps": [
                {"word": "أبي", "start": 0.08, "end": 0.2},
                {"word": "أحول", "start": 0.2, "end": 0.56},
                {"word": "درهم", "start": 0.56, "end": 1.56},
                {"word": "لفرحان", "start": 1.56, "end": 2.2},
            ],
            "stats": {"fileName": "audio.wav", "fileSize": "0.12 MB", "creditsConsumed": 4},
        },
        "message": "Success",
    }
    return web.json_response(body, status=200)


def make_app(state: FakeMunsitState) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_get("/api/v1/websocket/speech-to-text", _ws_handler)
    app.router.add_post("/api/v1/audio/transcribe", _batch_handler)
    return app


class FakeMunsitServer:
    """Convenience wrapper. Use via the ``fake_munsit`` pytest fixture."""

    def __init__(self) -> None:
        self.state = FakeMunsitState()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.url: str = ""
        self.batch_url: str = ""

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
        self.batch_url = f"http://127.0.0.1:{port}/api/v1/audio/transcribe"
        self.state.base_url = self.url
        return self.url

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    def script(self, entries: list[tuple[float, str, str]]) -> None:
        self.state.script = list(entries)
