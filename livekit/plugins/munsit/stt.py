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

from ._utils import AudioEnergyFilter, PeriodicCollector, build_wav_header
from .log import logger
from .models import MunsitModels

DEFAULT_BASE_URL = "wss://api.munsit.com/api/v1/listen"
DEFAULT_BATCH_BASE_URL = "https://api.munsit.com/api/v1/audio/transcribe"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ENDPOINTING_MS = 800
# Legacy defaults, kept for deprecation mapping:
DEFAULT_FINALIZE_AFTER_SILENCE_MS = 700
DEFAULT_VAD_SILENCE_MS = 1500

# Models the plugin accepts. We validate client-side because Munsit's API silently
# accepts arbitrary `model` query values (including empty strings and typos) and
# routes them to undefined fallback paths. To use a model not yet listed here,
# pass an explicit override via `extra_query_params={"model": "..."}`.
_VALID_MODELS: tuple[str, ...] = ("munsit", "munsit-en-ar")

AuthMethod = Literal["header", "bearer", "query"]
EndpointingMode = Literal["server_diff", "client_vad"]
Mode = Literal["batch", "streaming"]


def _build_batch_url_and_headers(opts: _STTOptions) -> tuple[str, dict[str, str]]:
    url = opts.batch_base_url
    headers: dict[str, str] = {}
    if opts.auth_method == "header":
        headers["x-api-key"] = opts.api_key
    elif opts.auth_method == "bearer":
        headers["Authorization"] = f"Bearer {opts.api_key}"
    else:  # "query"
        sep = "&" if "?" in url else "?"
        # send both spellings: `api_key` (current gateway) and `token` (legacy backend)
        url = f"{url}{sep}{urlencode({'api_key': opts.api_key, 'token': opts.api_key})}"
    return url, headers


async def _post_batch_audio(
    *,
    session: aiohttp.ClientSession,
    opts: _STTOptions,
    conn_options: APIConnectOptions,
    wav_bytes: bytes,
) -> dict:
    """Send a multipart POST to Munsit's batch transcribe endpoint.

    Shared by ``SpeechStream._submit_batch`` (per-utterance batching while
    streaming) and ``STT._recognize_impl`` (synchronous full-buffer recognition).
    """
    url, headers = _build_batch_url_and_headers(opts)
    form = aiohttp.FormData()
    form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", str(opts.model))
    if opts.hotwords and str(opts.model) != "munsit-en-ar":
        form.add_field("hotwords", ",".join(opts.hotwords))
    if opts.return_confidence:
        form.add_field("return_confidence", "true")
    try:
        async with session.post(
            url,
            data=form,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=conn_options.timeout * 6),
        ) as resp:
            body = await resp.text()
            if resp.status in (401, 403):
                raise APIStatusError(
                    message=f"Munsit auth rejected: {body[:200]}",
                    status_code=resp.status,
                    request_id=None,
                    body=body,
                )
            if resp.status >= 400:
                raise APIStatusError(
                    message=f"Munsit batch HTTP {resp.status}: {body[:200]}",
                    status_code=resp.status,
                    request_id=None,
                    body=body,
                )
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as e:
                raise APIConnectionError(
                    f"Munsit batch returned non-JSON body: {body[:200]}"
                ) from e
            # Munsit's batch shape: {"statusCode": 200, "data": {...}, "message": "Success"}
            data = parsed.get("data") if isinstance(parsed, dict) else None
            if not isinstance(data, dict):
                raise APIConnectionError(f"Munsit batch response missing 'data': {body[:200]}")
            return data
    except asyncio.TimeoutError as e:
        raise APITimeoutError() from e
    except aiohttp.ClientError as e:
        raise APIConnectionError(f"Munsit batch connection error: {e}") from e


def _batch_response_to_speech_data(
    opts: _STTOptions,
    text: str,
    duration: float,
    timestamps: list[dict[str, object]],
) -> stt.SpeechData:
    from livekit.agents.voice.io import TimedString

    words: list[TimedString] = []
    confidences: list[float] = []
    for ts in timestamps:
        word = ts.get("word")
        start = ts.get("start")
        end = ts.get("end")
        if (
            isinstance(word, str)
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
        ):
            words.append(TimedString(text=word, start_time=float(start), end_time=float(end)))
            conf = ts.get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
    return stt.SpeechData(
        language=LanguageCode(opts.language or "ar"),
        start_time=0.0,
        end_time=float(duration),
        confidence=(sum(confidences) / len(confidences)) if confidences else 1.0,
        text=text.strip(),
        words=words,
    )


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    batch_base_url: str
    mode: Mode
    model: MunsitModels | str
    auth_method: AuthMethod
    sample_rate: int
    num_channels: int
    interim_results: bool
    endpointing_ms: int
    smart_turn: bool
    hotwords: list[str] | None
    correlation_id: str | None
    return_confidence: bool
    language: str | None
    extra_query_params: dict[str, str] | None
    on_enrichment: object | None = None   # callable(dict) for Sentiment/Gender events


class STT(stt.STT):
    def __init__(
        self,
        *,
        mode: Mode = "streaming",
        model: MunsitModels | str = "munsit-en-ar",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        batch_base_url: NotGivenOr[str] = NOT_GIVEN,
        auth_method: AuthMethod = "header",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = 1,
        interim_results: bool = True,
        endpointing_ms: int = DEFAULT_ENDPOINTING_MS,
        smart_turn: bool = True,
        hotwords: list[str] | None = None,
        correlation_id: str | None = None,
        return_confidence: bool = False,
        language: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        extra_query_params: dict[str, str] | None = None,
        on_enrichment: object | None = None,
        # ── deprecated (0.3.x legacy socket) — accepted, mapped, warned ──
        endpointing: NotGivenOr[EndpointingMode] = NOT_GIVEN,
        finalize_after_silence_ms: NotGivenOr[int] = NOT_GIVEN,
        energy_filter: AudioEnergyFilter | bool = False,
        vad_silence_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Munsit STT.

        Args:
            mode: ``"streaming"`` (default) keeps a live WebSocket to ``/api/v1/listen``:
                interim results while speaking, server-side turn detection, word
                timestamps and confidence on finals. ``"batch"`` buffers audio per
                utterance and POSTs the WAV to ``/api/v1/audio/transcribe``.
            model: ``"munsit-en-ar"`` (default — Arabic + English code-switch) or
                ``"munsit"`` (pure Arabic, fastest). Applies to BOTH modes: batch sends
                it as a form field; streaming sends it on the socket query, where the
                server routes the session to the matching live engine.
            api_key: Munsit API key. Falls back to the ``MUNSIT_API_KEY`` env var.
            base_url: Override the streaming WebSocket URL (``mode="streaming"``).
            batch_base_url: Override the batch HTTP URL (``mode="batch"``).
            auth_method: ``"header"`` (``x-api-key``), ``"bearer"`` (``Authorization``), or
                ``"query"`` (sends ``api_key=`` — plus legacy ``token=`` on the batch URL).
            sample_rate: Input sample rate hint. Streaming negotiates 8000 or 16000 with the
                server and resamples anything else (LiveKit tracks are typically 48000).
            num_channels: Number of audio channels (batch WAV header).
            interim_results: Emit ``INTERIM_TRANSCRIPT`` events while a turn is in progress
                (streaming mode).
            endpointing_ms: Server-side silence window that ends a turn (100-5000, default
                800). Retunable live via ``ListenStream.configure_endpointing``.
            smart_turn: Gate end-of-turn on the server's semantic turn-completion model in
                addition to silence (default on).
            hotwords: Custom vocabulary — a list of rare terms/phrases to bias recognition
                toward (up to 200 entries of up to 40 characters). Applies to streaming and
                batch on the ``munsit`` model; ignored (with a warning) for
                ``munsit-en-ar``, which has no biasing.
            correlation_id: Session identifier echoed on every streaming event (<=128 chars).
            return_confidence: Batch mode — request per-word confidence in timestamps.
            language: Transcription language. ``"ar"`` (default) is the only streaming value.
            http_session: Custom aiohttp session. Falls back to the LiveKit shared session.
            extra_query_params: Forward-compat for new query params without an SDK update.
            on_enrichment: Optional callable receiving raw ``Sentiment``/``Gender`` event
                dicts from the streaming session (they have no LiveKit event equivalent).
            endpointing: DEPRECATED (legacy socket). Ignored; server-side endpointing is
                always used.
            finalize_after_silence_ms: DEPRECATED. Mapped to ``endpointing_ms`` if that was
                not set explicitly.
            energy_filter: DEPRECATED for streaming (batch keeps an internal energy VAD).
            vad_silence_ms: DEPRECATED. Ignored.
        """
        import warnings

        if is_given(endpointing):
            warnings.warn(
                "endpointing='server_diff'/'client_vad' is deprecated: /api/v1/listen does "
                "server-side turn detection. Use endpointing_ms and smart_turn instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if is_given(finalize_after_silence_ms):
            warnings.warn(
                "finalize_after_silence_ms is deprecated; use endpointing_ms.",
                DeprecationWarning,
                stacklevel=2,
            )
            if endpointing_ms == DEFAULT_ENDPOINTING_MS:
                endpointing_ms = max(100, min(5000, int(finalize_after_silence_ms)))
        if is_given(vad_silence_ms):
            warnings.warn("vad_silence_ms is deprecated and ignored.", DeprecationWarning,
                          stacklevel=2)
        if energy_filter is not False:
            warnings.warn(
                "energy_filter is deprecated: batch mode uses an internal energy VAD and "
                "streaming mode uses server-side turn detection.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=(mode == "streaming" and interim_results),
                # Both modes return word timestamps now.
                aligned_transcript="word",
            )
        )

        munsit_api_key = api_key if is_given(api_key) else os.environ.get("MUNSIT_API_KEY")
        if not munsit_api_key:
            raise ValueError(
                "Munsit API key is required, either as argument or via MUNSIT_API_KEY env var"
            )

        if mode not in ("batch", "streaming"):
            raise ValueError(f"mode must be 'batch' or 'streaming'; got {mode!r}")

        if str(model) not in _VALID_MODELS:
            raise ValueError(
                f"model must be one of {_VALID_MODELS}; got {model!r}. "
                "To use a model not yet listed here, pass an override via "
                "extra_query_params={'model': '...'}."
            )

        if auth_method not in ("header", "bearer", "query"):
            raise ValueError(
                f"auth_method must be 'header', 'bearer', or 'query'; got {auth_method!r}"
            )
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if not 100 <= endpointing_ms <= 5000:
            raise ValueError("endpointing_ms must be within 100..5000")
        if hotwords is not None:
            if len(hotwords) > 200:
                raise ValueError("hotwords supports at most 200 entries")
            for w in hotwords:
                if len(w) > 40:
                    raise ValueError(f"hotword entries must be <= 40 characters; got {w!r}")
                if "," in w:
                    raise ValueError(f"hotword entries must not contain commas; got {w!r}")
        if correlation_id is not None and len(correlation_id) > 128:
            raise ValueError("correlation_id must be <= 128 characters")
        if str(model) == "munsit-en-ar" and hotwords:
            # Mirror faseeh's server-side behavior (it silently drops biasing for this
            # model) instead of raising: with en-ar as the default model, a hard error
            # would break every existing `STT(hotwords=[...])` call site.
            warnings.warn(
                "hotwords are not supported with model 'munsit-en-ar' and will be "
                "ignored — recognition biasing exists only on the Arabic 'munsit' model",
                UserWarning,
                stacklevel=2,
            )
            hotwords = None

        self._opts = _STTOptions(
            api_key=munsit_api_key,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            batch_base_url=batch_base_url if is_given(batch_base_url) else DEFAULT_BATCH_BASE_URL,
            mode=mode,
            model=model,
            auth_method=auth_method,
            sample_rate=sample_rate,
            num_channels=num_channels,
            interim_results=interim_results,
            endpointing_ms=endpointing_ms,
            smart_turn=smart_turn,
            hotwords=list(hotwords) if hotwords else None,
            correlation_id=correlation_id,
            return_confidence=return_confidence,
            language=language,
            extra_query_params=extra_query_params,
            on_enrichment=on_enrichment,
        )
        if isinstance(energy_filter, AudioEnergyFilter):
            self._energy_filter: AudioEnergyFilter | None = energy_filter
        elif energy_filter is True:
            self._energy_filter = AudioEnergyFilter()
        else:
            self._energy_filter = None
        self._session = http_session
        self._streams: weakref.WeakSet[stt.SpeechStream] = weakref.WeakSet()

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
        endpointing_ms: int | None = None,
        smart_turn: bool | None = None,
        hotwords: list[str] | None = None,
    ) -> None:
        """Update STT options. Live streams reconnect to apply changes."""
        if model is not None:
            if str(model) not in _VALID_MODELS:
                raise ValueError(f"model must be one of {_VALID_MODELS}; got {model!r}.")
            self._opts.model = model
        if language is not None:
            self._opts.language = language
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if endpointing_ms is not None:
            if not 100 <= endpointing_ms <= 5000:
                raise ValueError("endpointing_ms must be within 100..5000")
            self._opts.endpointing_ms = endpointing_ms
        if smart_turn is not None:
            self._opts.smart_turn = smart_turn
        if hotwords is not None:
            self._opts.hotwords = list(hotwords)
        for stream in self._streams:
            for attr, val in (
                ("model", model),
                ("language", language),
                ("interim_results", interim_results),
                ("endpointing_ms", endpointing_ms),
                ("smart_turn", smart_turn),
                ("hotwords", list(hotwords) if hotwords is not None else None),
            ):
                if val is not None:
                    setattr(stream._opts, attr, val)
            stream._reconnect_event.set()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        speech_stream: stt.SpeechStream
        if opts.mode == "streaming":
            from ._listen import ListenStream

            speech_stream = ListenStream(
                stt_impl=self,
                opts=opts,
                conn_options=conn_options,
                http_session=self._ensure_session(),
            )
        else:
            speech_stream = SpeechStream(
                stt=self,
                opts=opts,
                conn_options=conn_options,
                http_session=self._ensure_session(),
                energy_filter=self._energy_filter,
            )
        self._streams.add(speech_stream)
        return speech_stream

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Synchronous batch transcription via Munsit's HTTP endpoint.

        Combines the AudioBuffer's frames into a single WAV blob and POSTs to
        ``batch_base_url``. Returns a single FINAL_TRANSCRIPT SpeechEvent with
        per-word timestamps populated. Works regardless of the configured
        ``mode`` — synchronous recognition always uses HTTP.
        """
        combined = rtc.combine_audio_frames(buffer)
        wav_bytes = build_wav_header(
            sample_rate=combined.sample_rate, num_channels=combined.num_channels
        ) + bytes(combined.data)

        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        data = await _post_batch_audio(
            session=self._ensure_session(),
            opts=opts,
            conn_options=conn_options,
            wav_bytes=wav_bytes,
        )

        text = data.get("transcription", "") or ""
        duration = data.get("duration") or 0.0
        timestamps = data.get("timestamps") or []
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(data.get("transcriptionId") or utils.shortuuid()),
            alternatives=[_batch_response_to_speech_data(opts, text, duration, timestamps)],
        )


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
        # Batch-mode state:
        self._batch_buffer: bytearray | None = None
        self._batch_sample_rate: int = opts.sample_rate
        self._batch_num_channels: int = opts.num_channels

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

    async def _run(self) -> None:
        # SpeechStream is the batch-mode implementation; mode="streaming" uses
        # ListenStream (see _listen.py) against WS /api/v1/listen.
        await self._run_batch()

    async def _run_batch(self) -> None:
        """Batch mode: buffer audio per utterance, POST WAV when speech ends.

        Uses an internal energy-based VAD to drive utterance segmentation.
        This is necessary because LiveKit's AgentSession does not call
        ``stream.flush()`` on the STT — it instead pushes silence frames after
        VAD detects end-of-speech and waits for the STT to emit a FINAL on
        its own. Streaming-mode STTs respond via their idle timer; batch
        STTs need to detect that silence themselves and submit.

        Triggers for submitting the buffer (in priority order):
          1. AudioEnergyFilter transitions to ``State.END`` after at least one
             non-silent frame was buffered. This is the normal AgentSession
             flow — VAD-detected end-of-speech, AgentSession pushes silence,
             our energy filter sees the silence and ends the utterance.
          2. ``_FlushSentinel`` on the input channel (explicit flush from
             a consumer that knows what it's doing).
          3. Input-channel close (consumer called ``end_input()`` or the
             session ended).

        First-frame-non-silent gating: ``_append_to_batch_buffer`` only starts
        a buffer when it sees a non-silent frame, so leading silence (before
        the user starts speaking) doesn't fire a spurious START_OF_SPEECH.
        """
        # 500 ms of trailing silence is short enough to keep EOU latency low
        # but long enough to ride out brief mid-utterance pauses.
        energy = AudioEnergyFilter(min_silence=0.5)

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                state = energy.update(data)
                if state in (
                    AudioEnergyFilter.State.START,
                    AudioEnergyFilter.State.SPEAKING,
                ):
                    self._append_to_batch_buffer(data)
                elif state == AudioEnergyFilter.State.END:
                    if self._batch_buffer is not None and len(self._batch_buffer) > 0:
                        await self._submit_batch()
                # State.SILENCE: discard (pre-speech or post-finalize silence).
            elif isinstance(data, self._FlushSentinel):
                if self._batch_buffer is not None and len(self._batch_buffer) > 0:
                    await self._submit_batch()
        # Input channel exhausted; submit any remaining buffered audio.
        if self._batch_buffer is not None and len(self._batch_buffer) > 0:
            await self._submit_batch()

    def _append_to_batch_buffer(self, frame: rtc.AudioFrame) -> None:
        if self._batch_buffer is None:
            # First frame of a new utterance.
            self._batch_buffer = bytearray()
            self._batch_sample_rate = frame.sample_rate
            self._batch_num_channels = frame.num_channels
            self._request_id = utils.shortuuid()
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH, request_id=self._request_id
                )
            )
        self._batch_buffer.extend(bytes(frame.data))
        self._ensure_usage_collector().push(frame.duration)

    async def _submit_batch(self) -> None:
        """Build a WAV from the buffered PCM, POST to Munsit, emit FINAL events."""
        assert self._batch_buffer is not None
        wav_bytes = build_wav_header(
            sample_rate=self._batch_sample_rate, num_channels=self._batch_num_channels
        ) + bytes(self._batch_buffer)

        try:
            response_data = await _post_batch_audio(
                session=self._session,
                opts=self._opts,
                conn_options=self._conn_options,
                wav_bytes=wav_bytes,
            )
        except Exception:
            # Reset buffer regardless of error so next utterance starts fresh.
            self._batch_buffer = None
            raise

        text = response_data.get("transcription", "") or ""
        duration = response_data.get("duration") or 0.0
        timestamps = response_data.get("timestamps") or []
        speech_data = _batch_response_to_speech_data(self._opts, text, duration, timestamps)

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[speech_data],
            )
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id)
        )
        self._batch_buffer = None
