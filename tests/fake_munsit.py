# Copyright 2026 LiveKit, Inc.
"""Local server emulating Munsit's APIs for tests.

WS side emulates ``/api/v1/listen`` (binary audio in, typed JSON events out,
JSON control frames); HTTP side emulates ``POST /api/v1/audio/transcribe``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web


def default_open_metadata(**overrides: Any) -> dict[str, Any]:
    ev = {
        "type": "Metadata",
        "session_id": "fake-session",
        "correlation_id": None,
        "model": "munsit-v2",
        "protocol_version": 1,
        "channels": 1,
        "sample_rate": 16000,
        "dropped_hotwords": [],
        "sentiment": "available",
        "gender": "available",
    }
    ev.update(overrides)
    return ev


def default_close_metadata(**overrides: Any) -> dict[str, Any]:
    ev = {
        "type": "Metadata",
        "session_id": "fake-session",
        "audio_seconds_billed": 3.2,
        "turn_count": 1,
    }
    ev.update(overrides)
    return ev


@dataclass
class FakeMunsitState:
    base_url: str = ""

    # ---- /api/v1/listen state ----
    listen_received_headers: dict[str, str] = field(default_factory=dict)
    listen_received_query: dict[str, str] = field(default_factory=dict)
    listen_binary_bytes: int = 0
    listen_binary_frames: int = 0
    listen_controls: list[dict[str, Any]] = field(default_factory=list)
    # Events sent right after the connection opens. None -> [default_open_metadata()].
    on_open_events: list[dict[str, Any]] | None = None
    # (min_total_bytes, event) pairs — each fires once when cumulative binary bytes reach
    # the threshold.
    on_audio_events: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    # Events sent when CloseStream arrives, before closing with code 1000.
    # None -> [default_close_metadata()].
    on_close_events: list[dict[str, Any]] | None = None
    require_api_key: str | None = None
    close_after_bytes: int | None = None  # abrupt server-side close after N binary bytes

    # ---- Batch endpoint state ----
    batch_received_count: int = 0
    batch_received_headers: dict[str, str] = field(default_factory=dict)
    batch_received_query: dict[str, str] = field(default_factory=dict)
    batch_received_models: list[str] = field(default_factory=list)
    batch_received_hotwords: list[str] = field(default_factory=list)
    batch_received_return_confidence: list[str] = field(default_factory=list)
    batch_received_audio_sizes: list[int] = field(default_factory=list)
    batch_response_status: int = 200
    batch_response_body: dict[str, Any] | None = None


def _provided_key(request: web.Request) -> str | None:
    auth = request.headers.get("Authorization")
    bearer = auth[7:].strip() if auth and auth.lower().startswith("bearer ") else None
    return (
        request.query.get("api_key")
        or request.query.get("x-api-key")
        or request.headers.get("x-api-key")
        or bearer
        or request.query.get("token")
    )


async def _listen_handler(request: web.Request) -> web.StreamResponse:
    state: FakeMunsitState = request.app["state"]
    state.listen_received_headers = dict(request.headers)
    state.listen_received_query = dict(request.query)

    if state.require_api_key is not None and _provided_key(request) != state.require_api_key:
        return web.Response(status=401, text="invalid api key")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    for ev in (
        state.on_open_events if state.on_open_events is not None else [default_open_metadata()]
    ):
        await ws.send_str(json.dumps(ev))

    fired: set[int] = set()
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                state.listen_binary_bytes += len(msg.data)
                state.listen_binary_frames += 1
                for i, (threshold, ev) in enumerate(state.on_audio_events):
                    if i not in fired and state.listen_binary_bytes >= threshold:
                        fired.add(i)
                        await ws.send_str(json.dumps(ev))
                if (
                    state.close_after_bytes is not None
                    and state.listen_binary_bytes >= state.close_after_bytes
                ):
                    await ws.close()
                    break
            elif msg.type == WSMsgType.TEXT:
                try:
                    ctrl = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                state.listen_controls.append(ctrl)
                if ctrl.get("type") == "CloseStream":
                    for ev in (
                        state.on_close_events
                        if state.on_close_events is not None
                        else [default_close_metadata()]
                    ):
                        await ws.send_str(json.dumps(ev))
                    await ws.close(code=1000)
                    break
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        pass
    return ws


async def _batch_handler(request: web.Request) -> web.Response:
    state: FakeMunsitState = request.app["state"]
    state.batch_received_count += 1
    state.batch_received_headers = dict(request.headers)
    state.batch_received_query = dict(request.query)

    if state.require_api_key is not None and _provided_key(request) != state.require_api_key:
        return web.json_response({"statusCode": 401, "message": "Unauthorized"}, status=401)

    reader = await request.multipart()
    seen_model = ""
    seen_hotwords = ""
    seen_return_conf = ""
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
        elif part.name == "hotwords":
            seen_hotwords = (await part.read(decode=False)).decode("utf-8")
        elif part.name == "return_confidence":
            seen_return_conf = (await part.read(decode=False)).decode("utf-8")
    state.batch_received_models.append(seen_model)
    state.batch_received_hotwords.append(seen_hotwords)
    state.batch_received_return_confidence.append(seen_return_conf)
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
    app.router.add_get("/api/v1/listen", _listen_handler)
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
        assert self._site._server and self._site._server.sockets  # type: ignore[attr-defined]
        port = self._site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
        self.url = f"ws://127.0.0.1:{port}/api/v1/listen"
        self.batch_url = f"http://127.0.0.1:{port}/api/v1/audio/transcribe"
        self.state.base_url = self.url
        return self.url

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
