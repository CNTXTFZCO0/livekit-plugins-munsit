# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Streaming client for Munsit's live transcription WebSocket (``/api/v1/listen``).

Protocol summary (full reference: https://docs.munsit.com/api-reference/stt-streaming):
  * connect with query params (encoding/sample_rate/language/endpointing/smart_turn/
    hotwords/...), authenticated via ``x-api-key`` header or ``api_key`` query param
  * send raw binary PCM frames; JSON text frames are control messages
    (``KeepAlive`` / ``Configure`` / ``CloseStream``)
  * receive typed JSON events: ``Metadata`` (open/close), ``SpeechStarted``,
    ``Results`` (interim/final with word timestamps + confidence), ``UtteranceEnd``,
    ``Sentiment``, ``Gender``, ``Error``
  * server handles endpointing and turn detection — the client never finalizes

Event mapping to LiveKit:
  SpeechStarted                       -> START_OF_SPEECH
  Results (is_final=false)            -> INTERIM_TRANSCRIPT
  Results (is_final=true)             -> FINAL_TRANSCRIPT (words + confidence attached)
  UtteranceEnd                        -> END_OF_SPEECH
  Results final with speech_final=false is a forced split during unbroken speech:
  FINAL_TRANSCRIPT is emitted but no END_OF_SPEECH (the speaker is still talking).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)

from ._utils import PeriodicCollector
from .log import logger

if TYPE_CHECKING:
    from .stt import _STTOptions

KEEPALIVE_INTERVAL_S = 5.0     # server closes after 12 s without audio or KeepAlive
_TARGET_RATES = (8000, 16000)  # the only sample rates /listen accepts


class ListenStream(stt.SpeechStream):
    """SpeechStream implementation backed by ``WS /api/v1/listen``."""

    def __init__(
        self,
        *,
        stt_impl: stt.STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt_impl, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._reconnect_event: asyncio.Event = asyncio.Event()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._request_id: str = utils.shortuuid()
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_input_rate: int | None = None
        self._target_rate: int = (
            opts.sample_rate if opts.sample_rate in _TARGET_RATES else 16000
        )
        self._usage_collector = PeriodicCollector[float](
            callback=self._on_audio_duration_report, duration=5.0
        )
        self._initial_backoff = 1.0
        self._max_backoff = 30.0
        self._last_connect_succeeded_at = 0.0

    # ── connection ───────────────────────────────────────────────────────────

    def _build_url_and_headers(self) -> tuple[str, dict[str, str]]:
        # The live protocol accepts only language="ar" in v1; "ar-en" is the BATCH spelling
        # of the code-switch selection and would fail the handshake with 4002. Engine
        # selection rides the `model` param instead, so normalize rather than reject.
        language = self._opts.language or "ar"
        if language == "ar-en":
            language = "ar"
        params: dict[str, str] = {
            "encoding": "linear16",
            "sample_rate": str(self._target_rate),
            "channels": "1",
            "language": language,
            "interim_results": "true" if self._opts.interim_results else "false",
            "endpointing": str(self._opts.endpointing_ms),
            "smart_turn": "true" if self._opts.smart_turn else "false",
            # Engine selection: faseeh's /api/v1/listen routes "munsit-en-ar" to the
            # code-switch live engine (resolveLiveEngine). faseeh consumes this param —
            # it is never forwarded to the engine itself.
            "model": str(self._opts.model),
        }
        # No biasing on the en-ar backend — the server would just report every entry
        # dropped; the constructor already warned when it discarded them.
        if self._opts.hotwords and str(self._opts.model) != "munsit-en-ar":
            params["hotwords"] = ",".join(self._opts.hotwords)
        if self._opts.correlation_id:
            params["correlation_id"] = self._opts.correlation_id
        if self._opts.extra_query_params:
            params.update(self._opts.extra_query_params)

        headers: dict[str, str] = {}
        if self._opts.auth_method == "header":
            headers["x-api-key"] = self._opts.api_key
        elif self._opts.auth_method == "bearer":
            headers["Authorization"] = f"Bearer {self._opts.api_key}"
        else:  # "query"
            params["api_key"] = self._opts.api_key
        return f"{self._opts.base_url}?{urlencode(params)}", headers

    async def _run(self) -> None:
        backoff = self._initial_backoff
        while True:
            try:
                await self._connect_and_run_once()
                break  # input channel drained; clean exit
            except APIStatusError:
                raise  # auth/policy rejections don't retry
            except (APIConnectionError, APITimeoutError) as e:
                if (
                    self._last_connect_succeeded_at > 0
                    and time.monotonic() - self._last_connect_succeeded_at > 10.0
                ):
                    backoff = self._initial_backoff
                logger.warning("Munsit /listen error (%s), reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)

    async def _connect_and_run_once(self) -> None:
        url, headers = self._build_url_and_headers()
        try:
            async with self._session.ws_connect(
                url,
                headers=headers,
                timeout=aiohttp.ClientWSTimeout(
                    ws_receive=self._conn_options.timeout * 5, ws_close=10
                ),
            ) as ws:
                self._ws = ws
                self._last_connect_succeeded_at = time.monotonic()

                send_task = asyncio.create_task(self._send_audio_task())
                recv_task = asyncio.create_task(self._recv_events_task())
                keepalive_task = asyncio.create_task(self._keepalive_task())
                reconnect_wait = asyncio.create_task(self._reconnect_event.wait())
                tasks = [send_task, recv_task, keepalive_task, reconnect_wait]
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if not task.cancelled():
                            task.result()
                    if self._reconnect_event.is_set():
                        self._reconnect_event.clear()
                        raise APIConnectionError("Munsit options changed, reconnecting")
                    if send_task in done:
                        # Input drained: CloseStream was sent; wait for the server to
                        # flush finals and close (recv_task exits on close Metadata).
                        await asyncio.wait_for(recv_task, timeout=30.0)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
                    self._ws = None
        except aiohttp.WSServerHandshakeError as e:
            if e.status in (401, 403):
                raise APIStatusError(
                    message=f"Munsit auth rejected: {e.message}",
                    status_code=e.status,
                    request_id=None,
                    body=None,
                ) from e
            raise APIConnectionError(f"Munsit /listen handshake failed: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Munsit /listen connection error: {e}") from e

    # ── sending ──────────────────────────────────────────────────────────────

    @staticmethod
    def _downmix(frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """Average interleaved channels into mono (16-bit PCM)."""
        import numpy as np

        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
        mono = samples.mean(axis=1).astype(np.int16)
        return rtc.AudioFrame(
            data=mono.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono),
        )

    def _resample(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        """Convert an arbitrary input frame to mono PCM at the negotiated rate.

        /listen accepts only 8000 or 16000 Hz; LiveKit tracks commonly run at 48000, so a
        stateful resampler (no per-frame edge artifacts) is required.
        """
        if frame.num_channels != 1:
            frame = self._downmix(frame)
        if frame.sample_rate == self._target_rate:
            return [frame]
        if self._resampler is None or self._resampler_input_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate, output_rate=self._target_rate, num_channels=1
            )
            self._resampler_input_rate = frame.sample_rate
        return list(self._resampler.push(frame))

    async def _send_audio_task(self) -> None:
        async for data in self._input_ch:
            ws = self._ws
            if ws is None or ws.closed:
                break
            if isinstance(data, rtc.AudioFrame):
                self._usage_collector.push(data.duration)
                for out in self._resample(data):
                    await ws.send_bytes(bytes(out.data))
            elif isinstance(data, self._FlushSentinel):
                # Server-side endpointing decides finality; nothing to flush.
                continue
        ws = self._ws
        if ws is not None and not ws.closed:
            if self._resampler is not None:
                for out in self._resampler.flush():
                    await ws.send_bytes(bytes(out.data))
            await ws.send_str(json.dumps({"type": "CloseStream"}))

    async def _keepalive_task(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_S)
            ws = self._ws
            if ws is None or ws.closed:
                return
            await ws.send_str(json.dumps({"type": "KeepAlive"}))

    # ── receiving ────────────────────────────────────────────────────────────

    async def _recv_events_task(self) -> None:
        ws = self._ws
        if ws is None:
            return
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                continue
            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("Munsit /listen sent non-JSON text frame: %r", msg.data[:100])
                continue
            if self._handle_event(event):
                return  # close Metadata received — session complete
        raise APIConnectionError("Munsit /listen closed unexpectedly")

    def _handle_event(self, event: dict) -> bool:
        etype = event.get("type")
        if etype == "Metadata":
            if "audio_seconds_billed" in event:
                logger.debug(
                    "Munsit session %s closed: %.1fs billed, %d turns",
                    event.get("session_id"),
                    event.get("audio_seconds_billed", 0.0),
                    event.get("turn_count", 0),
                )
                return True
            logger.debug(
                "Munsit session %s open (model=%s, dropped_hotwords=%s)",
                event.get("session_id"),
                event.get("model"),
                event.get("dropped_hotwords"),
            )
            dropped = event.get("dropped_hotwords") or []
            if dropped:
                logger.warning("Munsit dropped hotword entries: %s", dropped)
        elif etype == "SpeechStarted":
            self._request_id = utils.shortuuid()
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH, request_id=self._request_id
                )
            )
        elif etype == "Results":
            self._handle_results(event)
        elif etype == "UtteranceEnd":
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                )
            )
        elif etype in ("Sentiment", "Gender"):
            # Not representable as LiveKit SpeechEvents; surfaced via the optional
            # callback for voice-analytics consumers.
            if self._opts.on_enrichment is not None:
                try:
                    self._opts.on_enrichment(event)
                except Exception:
                    logger.exception("Munsit on_enrichment callback raised")
        elif etype == "Error":
            code = event.get("code")
            message = event.get("message", "")
            if event.get("recoverable"):
                logger.warning("Munsit recoverable error %s: %s", code, message)
            else:
                raise APIStatusError(
                    message=f"Munsit /listen error {code}: {message}",
                    status_code=int(code) if isinstance(code, int) else 500,
                    request_id=self._request_id,
                    body=None,
                )
        return False

    def _handle_results(self, event: dict) -> None:
        text = event.get("transcript", "") or ""
        is_final = bool(event.get("is_final"))
        if not text and not is_final:
            return

        from livekit.agents.voice.io import TimedString

        words: list[TimedString] = []
        for w in event.get("words") or []:
            word = w.get("word")
            start = w.get("start")
            end = w.get("end")
            if (
                isinstance(word, str)
                and isinstance(start, (int, float))
                and isinstance(end, (int, float))
            ):
                words.append(TimedString(text=word, start_time=float(start), end_time=float(end)))

        confidence = event.get("confidence")
        data = stt.SpeechData(
            language=LanguageCode(event.get("language") or self._opts.language or "ar"),
            start_time=float(words[0].start_time) if words else 0.0,
            end_time=float(words[-1].end_time) if words else 0.0,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else 1.0,
            text=text,
            words=words,
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=(
                    stt.SpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT
                ),
                request_id=self._request_id,
                alternatives=[data],
            )
        )

    # ── misc ─────────────────────────────────────────────────────────────────

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    async def configure_endpointing(self, endpointing_ms: int) -> None:
        """Retune the server-side endpointing silence window mid-session (100-5000 ms)."""
        if not 100 <= endpointing_ms <= 5000:
            raise ValueError("endpointing_ms must be within 100..5000")
        self._opts.endpointing_ms = endpointing_ms
        ws = self._ws
        if ws is not None and not ws.closed:
            await ws.send_str(json.dumps({"type": "Configure", "endpointing": endpointing_ms}))
