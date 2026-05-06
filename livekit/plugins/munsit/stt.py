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

from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.utils import is_given

from ._utils import AudioEnergyFilter, PeriodicCollector, build_wav_header, pcm_to_audiobuffer
from .log import logger
from .models import MunsitModels

DEFAULT_BASE_URL = "wss://api.munsit.com/api/v1/websocket/speech-to-text"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FINALIZE_AFTER_SILENCE_MS = 700
DEFAULT_VAD_SILENCE_MS = 1500

AuthMethod = Literal["header", "bearer", "query"]
EndpointingMode = Literal["server_diff", "client_vad"]


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    model: MunsitModels | str
    auth_method: AuthMethod
    sample_rate: int
    num_channels: int
    interim_results: bool
    endpointing: EndpointingMode
    finalize_after_silence_ms: int
    vad_silence_ms: int
    language: str | None
    extra_query_params: dict[str, str] | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: MunsitModels | str = "munsit",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        auth_method: AuthMethod = "header",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = 1,
        interim_results: bool = True,
        endpointing: EndpointingMode = "server_diff",
        finalize_after_silence_ms: int = DEFAULT_FINALIZE_AFTER_SILENCE_MS,
        energy_filter: AudioEnergyFilter | bool = False,
        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
        language: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        extra_query_params: dict[str, str] | None = None,
    ) -> None:
        """Create a new instance of Munsit STT.

        Args:
            model: ASR model. ``"munsit"`` (default, Arabic) or ``"munsit-en-ar"`` (code-switching).
            api_key: Munsit API key. Falls back to the ``MUNSIT_API_KEY`` env var.
            base_url: Override the WebSocket URL (e.g. for staging or self-hosted Munsit).
            auth_method: How to send the API key. ``"header"`` (``x-api-key``), ``"bearer"``
                (``Authorization: Bearer ...``), or ``"query"`` (``?token=...``).
            sample_rate: Audio sample rate in Hz; used to synthesize the first-chunk WAV header.
            num_channels: Number of audio channels.
            interim_results: Emit ``INTERIM_TRANSCRIPT`` events as cumulative updates arrive.
            endpointing: ``"server_diff"`` (default) keeps a single WS open and finalizes after
                ``finalize_after_silence_ms`` of server silence. ``"client_vad"`` opens/closes the
                WS per utterance using a local energy filter.
            finalize_after_silence_ms: Idle threshold for ``server_diff`` finalization (>= 100).
            energy_filter: ``AudioEnergyFilter`` instance or ``True`` to use defaults; only
                consulted in ``"client_vad"`` mode. (Wired up in a later task.)
            vad_silence_ms: Silence duration that triggers utterance end in ``"client_vad"``
                (>= 100).
            language: Label attached to emitted ``SpeechData.language``. Defaults to
                ``"ar"`` if unset.
            http_session: Custom aiohttp session. Falls back to the LiveKit per-job shared session.
            extra_query_params: Forward-compat for new Munsit query params without an SDK update.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                aligned_transcript=False,
            )
        )

        munsit_api_key = api_key if is_given(api_key) else os.environ.get("MUNSIT_API_KEY")
        if not munsit_api_key:
            raise ValueError(
                "Munsit API key is required, either as argument or via MUNSIT_API_KEY env var"
            )

        if auth_method not in ("header", "bearer", "query"):
            raise ValueError(
                f"auth_method must be 'header', 'bearer', or 'query'; got {auth_method!r}"
            )
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if finalize_after_silence_ms < 100:
            raise ValueError("finalize_after_silence_ms must be >= 100")
        if vad_silence_ms < 100:
            raise ValueError("vad_silence_ms must be >= 100")

        self._opts = _STTOptions(
            api_key=munsit_api_key,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            model=model,
            auth_method=auth_method,
            sample_rate=sample_rate,
            num_channels=num_channels,
            interim_results=interim_results,
            endpointing=endpointing,
            finalize_after_silence_ms=finalize_after_silence_ms,
            vad_silence_ms=vad_silence_ms,
            language=language,
            extra_query_params=extra_query_params,
        )
        if isinstance(energy_filter, AudioEnergyFilter):
            self._energy_filter: AudioEnergyFilter | None = energy_filter
        elif energy_filter is True:
            self._energy_filter = AudioEnergyFilter(min_silence=vad_silence_ms / 1000.0)
        else:
            self._energy_filter = None
        self._session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "Munsit"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: MunsitModels | str | None = None,
        language: str | None = None,
        interim_results: bool | None = None,
        endpointing: EndpointingMode | None = None,
        finalize_after_silence_ms: int | None = None,
        vad_silence_ms: int | None = None,
    ) -> None:
        """Update STT options. Live streams reconnect to apply changes."""
        if model is not None:
            self._opts.model = model
        if language is not None:
            self._opts.language = language
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if endpointing is not None:
            self._opts.endpointing = endpointing
        if finalize_after_silence_ms is not None:
            if finalize_after_silence_ms < 100:
                raise ValueError("finalize_after_silence_ms must be >= 100")
            self._opts.finalize_after_silence_ms = finalize_after_silence_ms
        if vad_silence_ms is not None:
            if vad_silence_ms < 100:
                raise ValueError("vad_silence_ms must be >= 100")
            self._opts.vad_silence_ms = vad_silence_ms
        for stream in self._streams:
            # Propagate changed options into the stream's own copy so the next
            # WS connection (triggered by _reconnect_event) uses the new values.
            if model is not None:
                stream._opts.model = model
            if language is not None:
                stream._opts.language = language
            if interim_results is not None:
                stream._opts.interim_results = interim_results
            if endpointing is not None:
                stream._opts.endpointing = endpointing
            if finalize_after_silence_ms is not None:
                stream._opts.finalize_after_silence_ms = finalize_after_silence_ms
            if vad_silence_ms is not None:
                stream._opts.vad_silence_ms = vad_silence_ms
            stream._reconnect_event.set()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        speech_stream = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            http_session=self._ensure_session(),
            energy_filter=self._energy_filter,
        )
        self._streams.add(speech_stream)
        return speech_stream

    async def _recognize_impl(  # type: ignore[override]
        self, *args: object, **kwargs: object
    ) -> None:
        raise NotImplementedError("Munsit only supports streaming recognition")


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
        energy_filter: AudioEnergyFilter | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._reconnect_event: asyncio.Event = asyncio.Event()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._request_id: str = ""
        self._last_cumulative: str = ""
        self._last_msg_at: float = 0.0
        self._speaking: bool = False
        self._first_chunk_sent: bool = False
        self._initial_backoff: float = 1.0
        self._max_backoff: float = 30.0
        self._last_connect_succeeded_at: float = 0.0
        self._usage_report_interval: float = 5.0
        self._usage_collector: PeriodicCollector[float] | None = None
        self._energy_filter = energy_filter
        self._vad_state_active: bool = False

    def _ensure_usage_collector(self) -> PeriodicCollector[float]:
        if self._usage_collector is None:
            self._usage_collector = PeriodicCollector(
                callback=self._on_audio_duration_report,
                duration=self._usage_report_interval,
            )
        return self._usage_collector

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _build_url_and_headers(self) -> tuple[str, dict[str, str]]:
        params: dict[str, str] = {"model": str(self._opts.model)}
        if self._opts.extra_query_params:
            params.update(self._opts.extra_query_params)
        headers: dict[str, str] = {}
        if self._opts.auth_method == "header":
            headers["x-api-key"] = self._opts.api_key
        elif self._opts.auth_method == "bearer":
            headers["Authorization"] = f"Bearer {self._opts.api_key}"
        else:  # "query"
            params["token"] = self._opts.api_key
        url = f"{self._opts.base_url}?{urlencode(params)}"
        return url, headers

    async def _run(self) -> None:
        if self._opts.endpointing == "client_vad":
            await self._run_client_vad()
            return
        # server_diff (existing path):
        backoff = self._initial_backoff
        while True:
            try:
                await self._connect_and_run_once()
                # Clean exit (input channel closed) — break the reconnect loop.
                break
            except APIStatusError:
                # Auth failures (401/403) and application-level API errors propagate
                # immediately without retry; connection-level 5xx errors from the WS
                # handshake are converted to APIConnectionError in _connect_and_run_once.
                raise
            except (APIConnectionError, APITimeoutError) as e:
                # If the connection was healthy long enough (> 10 s), treat the next
                # failure as transient and restart backoff from the initial value.
                if (
                    self._last_connect_succeeded_at > 0
                    and time.monotonic() - self._last_connect_succeeded_at > 10.0
                ):
                    backoff = self._initial_backoff
                logger.warning(
                    "Munsit connection error (%s), reconnecting in %.1fs", e, backoff
                )
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
                self._request_id = utils.shortuuid()
                self._first_chunk_sent = False
                self._last_cumulative = ""
                self._speaking = False

                send_task = asyncio.create_task(self._send_audio_task())
                recv_task = asyncio.create_task(self._recv_messages_task())
                idle_task: asyncio.Task[None] | None = None
                if self._opts.endpointing == "server_diff":
                    idle_task = asyncio.create_task(self._idle_finalize_task())
                reconnect_wait = asyncio.create_task(self._reconnect_event.wait())
                tasks: list[asyncio.Task] = [send_task, recv_task, reconnect_wait]
                if idle_task is not None:
                    tasks.append(idle_task)
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if not task.cancelled():
                            task.result()
                    if self._reconnect_event.is_set():
                        self._reconnect_event.clear()
                        # Surface as a clean connection error so the outer loop reconnects.
                        raise APIConnectionError("Munsit options changed, reconnecting")
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
            raise APIConnectionError(f"Munsit handshake failed: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Munsit connection error: {e}") from e

    async def _run_client_vad(self) -> None:
        # Lazy default if energy_filter wasn't passed but mode is client_vad
        if self._energy_filter is None:
            self._energy_filter = AudioEnergyFilter(min_silence=self._opts.vad_silence_ms / 1000.0)

        utterance_recv_task: asyncio.Task[None] | None = None

        async def _close_current_utterance() -> None:
            nonlocal utterance_recv_task
            if self._last_cumulative and self._speaking:
                self._emit_final(self._last_cumulative)
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
            if utterance_recv_task is not None:
                try:
                    await utterance_recv_task
                except (asyncio.CancelledError, APIConnectionError):
                    pass
                except Exception as e:
                    logger.warning("Munsit utterance recv task error: %s", e)
                utterance_recv_task = None

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                state = self._energy_filter.update(data)
                if state == AudioEnergyFilter.State.START and not self._vad_state_active:
                    # Open a fresh WS for this utterance.
                    try:
                        url, headers = self._build_url_and_headers()
                        self._ws = await self._session.ws_connect(
                            url,
                            headers=headers,
                            timeout=aiohttp.ClientWSTimeout(
                                ws_receive=self._conn_options.timeout * 5, ws_close=10
                            ),
                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.warning(
                            "Munsit ws_connect failed in client_vad (%s); skipping utterance", e
                        )
                        self._ws = None
                        continue
                    self._request_id = utils.shortuuid()
                    self._first_chunk_sent = False
                    self._last_cumulative = ""
                    self._speaking = False
                    utterance_recv_task = asyncio.create_task(self._recv_messages_task())
                    self._vad_state_active = True
                    await self._send_audio_frame(data)
                elif state == AudioEnergyFilter.State.SPEAKING and self._vad_state_active:
                    await self._send_audio_frame(data)
                elif state == AudioEnergyFilter.State.END and self._vad_state_active:
                    await _close_current_utterance()
                    self._vad_state_active = False
            elif isinstance(data, self._FlushSentinel):
                if self._vad_state_active:
                    await _close_current_utterance()
                    self._vad_state_active = False

        if self._vad_state_active:
            await _close_current_utterance()
            self._vad_state_active = False

    async def _send_audio_task(self) -> None:
        if not self._ws:
            return
        async for data in self._input_ch:
            if not self._ws or self._ws.closed:
                break
            if isinstance(data, rtc.AudioFrame):
                await self._send_audio_frame(data)
            elif isinstance(data, self._FlushSentinel):
                if self._last_cumulative and self._speaking:
                    self._emit_final(self._last_cumulative)

    async def _send_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._ws:
            return
        pcm = bytes(frame.data)
        self._ensure_usage_collector().push(frame.duration)
        if not self._first_chunk_sent:
            # Build the WAV header from the actual frame metadata, not the constructor
            # defaults. AgentSession-driven frames will match self._opts; file-based
            # demos may have different sample rates (e.g. a 24kHz WAV) and the header
            # must declare what's really on the wire or Munsit will mis-decode.
            if frame.sample_rate != self._opts.sample_rate:
                logger.info(
                    "Munsit first frame sample_rate=%d differs from configured %d; "
                    "using frame's rate in WAV header.",
                    frame.sample_rate,
                    self._opts.sample_rate,
                )
            header = build_wav_header(
                sample_rate=frame.sample_rate, num_channels=frame.num_channels
            )
            payload = header + pcm
            self._first_chunk_sent = True
        else:
            payload = pcm
        message = {"event": "audio_chunk", "data": {"audioBuffer": pcm_to_audiobuffer(payload)}}
        await self._ws.send_str(json.dumps(message))

    async def _recv_messages_task(self) -> None:
        if not self._ws:
            return
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("Munsit sent non-JSON text frame: %r", msg.data[:100])
                    continue
                event = data.get("event") or data.get("type")
                payload = data.get("data") if data.get("data") is not None else data.get("text", "")
                if event == "transcription":
                    self._handle_transcription(payload if isinstance(payload, str) else "")
                elif event == "transcription_error":
                    err_text = payload if isinstance(payload, str) else json.dumps(payload)
                    raise APIStatusError(
                        message=f"Munsit transcription_error: {err_text}",
                        status_code=500,
                        request_id=self._request_id,
                        body=None,
                    )
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                break
        # The async-for loop exited, meaning the server closed the WS connection.
        # Treat this as a recoverable drop so the outer reconnect loop can re-connect.
        raise APIConnectionError("Munsit WS closed unexpectedly")

    def _handle_transcription(self, cumulative: str) -> None:
        if cumulative == self._last_cumulative:
            return
        self._last_cumulative = cumulative
        self._last_msg_at = time.monotonic()
        if not cumulative:
            return
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH, request_id=self._request_id
                )
            )
        if self._opts.interim_results:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[self._make_speech_data(cumulative)],
                )
            )

    def _make_speech_data(self, text: str) -> stt.SpeechData:
        lang = self._opts.language or "ar"
        return stt.SpeechData(
            language=LanguageCode(lang),
            start_time=0.0,
            end_time=0.0,
            confidence=1.0,
            text=text,
            words=[],
        )

    async def _idle_finalize_task(self) -> None:
        threshold = self._opts.finalize_after_silence_ms / 1000.0
        while True:
            await asyncio.sleep(0.05)
            if not self._speaking or not self._last_cumulative:
                continue
            if (time.monotonic() - self._last_msg_at) >= threshold:
                self._emit_final(self._last_cumulative)

    def _emit_final(self, text: str) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[self._make_speech_data(text)],
            )
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
            )
        )
        self._last_cumulative = ""
        self._speaking = False
