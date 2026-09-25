# Copyright 2026 LiveKit, Inc.
import asyncio
import os
import pathlib
import struct
import wave
from unittest.mock import patch

import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.agents.stt import SpeechEventType
from livekit.plugins.munsit import STT
from livekit.plugins.munsit._utils import build_wav_header, pcm_to_audiobuffer


class TestWavHeader:
    def test_returns_44_bytes(self):
        header = build_wav_header(sample_rate=16000, num_channels=1)
        assert len(header) == 44

    def test_riff_and_wave_markers(self):
        header = build_wav_header(sample_rate=16000, num_channels=1)
        assert header[0:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        assert header[12:16] == b"fmt "
        assert header[36:40] == b"data"

    def test_fmt_chunk_pcm16_mono_16khz(self):
        header = build_wav_header(sample_rate=16000, num_channels=1)
        fmt_chunk_size = struct.unpack_from("<I", header, 16)[0]
        audio_format = struct.unpack_from("<H", header, 20)[0]
        channels = struct.unpack_from("<H", header, 22)[0]
        rate = struct.unpack_from("<I", header, 24)[0]
        byte_rate = struct.unpack_from("<I", header, 28)[0]
        block_align = struct.unpack_from("<H", header, 32)[0]
        bits_per_sample = struct.unpack_from("<H", header, 34)[0]
        assert fmt_chunk_size == 16
        assert audio_format == 1
        assert channels == 1
        assert rate == 16000
        assert bits_per_sample == 16
        assert block_align == channels * (bits_per_sample // 8)
        assert byte_rate == rate * block_align

    def test_stereo_24khz(self):
        header = build_wav_header(sample_rate=24000, num_channels=2)
        rate = struct.unpack_from("<I", header, 24)[0]
        channels = struct.unpack_from("<H", header, 22)[0]
        assert rate == 24000
        assert channels == 2

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            build_wav_header(sample_rate=0)

    def test_invalid_num_channels_raises(self):
        with pytest.raises(ValueError, match="num_channels"):
            build_wav_header(sample_rate=16000, num_channels=0)

    def test_invalid_bits_per_sample_raises(self):
        with pytest.raises(ValueError, match="bits_per_sample"):
            build_wav_header(sample_rate=16000, bits_per_sample=12)


class TestPcmToAudiobuffer:
    def test_converts_bytes_to_int_list(self):
        assert pcm_to_audiobuffer(b"\x00\x01\xff") == [0, 1, 255]

    def test_empty_bytes(self):
        assert pcm_to_audiobuffer(b"") == []


class TestSTTConstructor:
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MUNSIT_API_KEY"):
                STT()

    def test_reads_env_var(self):
        with patch.dict(os.environ, {"MUNSIT_API_KEY": "env-key"}, clear=True):
            stt = STT()
            assert stt._opts.api_key == "env-key"

    def test_explicit_arg_wins_over_env(self):
        with patch.dict(os.environ, {"MUNSIT_API_KEY": "env-key"}, clear=True):
            stt = STT(api_key="arg-key")
            assert stt._opts.api_key == "arg-key"

    def test_default_model(self):
        stt = STT(api_key="x")
        assert stt.model == "munsit-en-ar"
        assert stt.provider == "Munsit"

    def test_custom_model(self):
        stt = STT(api_key="x", model="munsit")
        assert stt.model == "munsit"

    def test_invalid_auth_method(self):
        with pytest.raises(ValueError, match="auth_method"):
            STT(api_key="x", auth_method="wrong")  # type: ignore[arg-type]

    def test_invalid_model_at_init_raises(self):
        with pytest.raises(ValueError, match="model"):
            STT(api_key="x", model="munsit-en-")  # typo

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            STT(api_key="x", model="")

    def test_garbage_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            STT(api_key="x", model="definitely-not-a-real-model-xyz")

    def test_update_options_invalid_model_raises(self):
        stt = STT(api_key="x")
        with pytest.raises(ValueError, match="model"):
            stt.update_options(model="not-a-real-model")

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            STT(api_key="x", sample_rate=0)

    def test_invalid_endpointing_ms(self):
        with pytest.raises(ValueError, match="endpointing_ms"):
            STT(api_key="x", endpointing_ms=50)
        with pytest.raises(ValueError, match="endpointing_ms"):
            STT(api_key="x", endpointing_ms=6000)

    def test_hotword_limits(self):
        with pytest.raises(ValueError, match="200"):
            STT(api_key="x", hotwords=[f"w{i}" for i in range(201)])
        with pytest.raises(ValueError, match="40"):
            STT(api_key="x", hotwords=["x" * 41])
        with pytest.raises(ValueError, match="comma"):
            STT(api_key="x", hotwords=["a,b"])

    def test_hotwords_dropped_with_warning_for_en_ar(self):
        with pytest.warns(UserWarning, match="munsit-en-ar"):
            stt = STT(api_key="x", model="munsit-en-ar", hotwords=["term"])
        assert stt._opts.hotwords is None

    def test_correlation_id_length(self):
        with pytest.raises(ValueError, match="correlation_id"):
            STT(api_key="x", correlation_id="c" * 129)

    def test_deprecated_finalize_maps_to_endpointing(self):
        with pytest.warns(DeprecationWarning, match="finalize_after_silence_ms"):
            stt = STT(api_key="x", finalize_after_silence_ms=700)
        assert stt._opts.endpointing_ms == 700

    def test_deprecated_endpointing_mode_warns(self):
        with pytest.warns(DeprecationWarning, match="server-side"):
            STT(api_key="x", endpointing="client_vad")

    def test_deprecated_vad_silence_warns(self):
        with pytest.warns(DeprecationWarning, match="vad_silence_ms"):
            STT(api_key="x", vad_silence_ms=1500)

    def test_default_mode_is_streaming(self):
        stt = STT(api_key="x")
        assert stt._opts.mode == "streaming"
        assert stt.capabilities.interim_results is True

    def test_capabilities_batch(self):
        stt = STT(api_key="x", mode="batch")
        assert stt.capabilities.streaming is True
        # batch mode never emits interims regardless of the flag
        assert stt.capabilities.interim_results is False
        # batch returns word timestamps from the HTTP response
        assert stt.capabilities.aligned_transcript == "word"

    def test_capabilities_streaming(self):
        stt = STT(api_key="x", mode="streaming")
        assert stt.capabilities.streaming is True
        assert stt.capabilities.interim_results is True
        # /listen finals carry word timestamps
        assert stt.capabilities.aligned_transcript == "word"

    def test_update_options_changes_model(self):
        stt = STT(api_key="x")
        stt.update_options(model="munsit-en-ar")
        assert stt.model == "munsit-en-ar"

    def test_capabilities_with_interim_disabled_in_streaming(self):
        stt = STT(api_key="x", mode="streaming", interim_results=False)
        assert stt.capabilities.interim_results is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            STT(api_key="x", mode="not-a-mode")  # type: ignore[arg-type]

    def test_update_options_invalid_endpointing(self):
        stt = STT(api_key="x")
        with pytest.raises(ValueError, match="endpointing_ms"):
            stt.update_options(endpointing_ms=50)


def _silence_frame(duration_ms: int = 100, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * duration_ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _collect_events_to_queue(stream: object, q: asyncio.Queue) -> None:  # type: ignore[type-arg]
    """Background helper: drain stream events into a queue so tests can poll with timeouts
    without cancelling the underlying async generator (which would corrupt the tee_peer state)."""
    async for ev in stream:  # type: ignore[union-attr]
        await q.put(ev)


def _loud_frame(duration_ms: int = 100, sample_rate: int = 16000) -> rtc.AudioFrame:
    """Generate a frame with int16 amplitude ~16384 (loud) so the energy filter triggers."""
    import array

    samples = sample_rate * duration_ms // 1000
    arr = array.array("h", [16000] * samples)
    return rtc.AudioFrame(
        data=arr.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


# ---------------------------------------------------------------------------
# Streaming mode tests — exercise the WS /api/v1/listen path (ListenStream).
# ---------------------------------------------------------------------------
def _final_results_event(**overrides):
    ev = {
        "type": "Results",
        "channel": 0,
        "turn_id": 0,
        "transcript": "لا تتكلم هكذا",
        "words": [
            {"word": "لا", "start": 0.1, "end": 0.3, "confidence": 0.99},
            {"word": "تتكلم", "start": 0.3, "end": 0.7, "confidence": 0.97},
            {"word": "هكذا", "start": 0.7, "end": 1.1, "confidence": 0.98},
        ],
        "is_final": True,
        "speech_final": True,
        "language": "ar",
        "confidence": 0.98,
    }
    ev.update(overrides)
    return ev


async def _drive_listen_stream(stream, frames, q, max_wait=8.0, stop_type=None):
    """Push frames, end input, and collect events until stop_type (or channel close)."""
    events = []
    collector = asyncio.create_task(_collect_events_to_queue(stream, q))
    try:
        for f in frames:
            stream.push_frame(f)
            await asyncio.sleep(0.01)
        stream.end_input()
        deadline = asyncio.get_event_loop().time() + max_wait
        while asyncio.get_event_loop().time() < deadline:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                if collector.done():
                    break
                continue
            events.append(ev)
            if stop_type is not None and ev.type == stop_type:
                break
    finally:
        collector.cancel()
        await stream.aclose()
    return events


class TestListenStreamEvents:
    async def test_full_turn_event_mapping(self, fake_munsit, http_session):
        """SpeechStarted -> START, interim -> INTERIM, final -> FINAL (words+conf),
        UtteranceEnd -> END_OF_SPEECH."""
        fake_munsit.state.on_audio_events = [
            (1, {"type": "SpeechStarted", "channel": 0, "ts": 0.5}),
            (
                1,
                {
                    "type": "Results",
                    "channel": 0,
                    "turn_id": 0,
                    "transcript": "لا",
                    "words": [{"word": "لا", "start": 0.1, "end": 0.3, "confidence": 1.0}],
                    "is_final": False,
                    "speech_final": False,
                    "language": "ar",
                    "confidence": None,
                },
            ),
            (1, _final_results_event()),
            (1, {"type": "UtteranceEnd", "channel": 0, "turn_id": 0, "last_word_end": 1.1}),
        ]
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        events = await _drive_listen_stream(
            stream,
            [_silence_frame() for _ in range(3)],
            q,
            stop_type=SpeechEventType.END_OF_SPEECH,
        )

        types = [e.type for e in events]
        assert SpeechEventType.START_OF_SPEECH in types
        assert SpeechEventType.INTERIM_TRANSCRIPT in types
        assert SpeechEventType.FINAL_TRANSCRIPT in types
        assert SpeechEventType.END_OF_SPEECH in types

        final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
        sd = final.alternatives[0]
        assert sd.text == "لا تتكلم هكذا"
        assert len(sd.words) == 3
        assert sd.words[0].start_time == 0.1
        assert abs(sd.confidence - 0.98) < 1e-6

    async def test_forced_split_emits_final_without_end_of_speech(self, fake_munsit, http_session):
        fake_munsit.state.on_audio_events = [
            (1, _final_results_event(speech_final=False)),
        ]
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        events = await _drive_listen_stream(stream, [_silence_frame()], q, max_wait=4.0)

        types = [e.type for e in events]
        assert SpeechEventType.FINAL_TRANSCRIPT in types
        assert SpeechEventType.END_OF_SPEECH not in types

    async def test_enrichment_callback_receives_sentiment_and_gender(
        self, fake_munsit, http_session
    ):
        fake_munsit.state.on_audio_events = [
            (1, _final_results_event()),
            (1, {"type": "UtteranceEnd", "channel": 0, "turn_id": 0, "last_word_end": 1.1}),
            (1, {"type": "Gender", "channel": 0, "turn_id": 0, "label": "female", "score": 0.99}),
            (
                1,
                {
                    "type": "Sentiment",
                    "channel": 0,
                    "turn_id": 0,
                    "label": "negative",
                    "score": 0.9,
                },
            ),
        ]
        enriched: list = []
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            http_session=http_session,
            on_enrichment=enriched.append,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        await _drive_listen_stream(stream, [_silence_frame()], q, max_wait=4.0)

        kinds = {e.get("type") for e in enriched}
        assert kinds == {"Gender", "Sentiment"}

    async def test_recoverable_error_does_not_raise(self, fake_munsit, http_session):
        fake_munsit.state.on_audio_events = [
            (1, {"type": "Error", "code": 4002, "message": "bad configure", "recoverable": True}),
            (1, _final_results_event()),
        ]
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        events = await _drive_listen_stream(
            stream,
            [_silence_frame()],
            q,
            stop_type=SpeechEventType.FINAL_TRANSCRIPT,
            max_wait=4.0,
        )
        assert any(e.type == SpeechEventType.FINAL_TRANSCRIPT for e in events)


class TestListenConnection:
    async def test_query_params_and_binary_audio(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="streaming",
            model="munsit",  # hotwords only exist on the Arabic model
            api_key="k",
            base_url=fake_munsit.url,
            http_session=http_session,
            endpointing_ms=300,
            smart_turn=False,
            hotwords=["عبد القادر", "أديب"],
            correlation_id="call-1",
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        await _drive_listen_stream(stream, [_silence_frame() for _ in range(2)], q, max_wait=3.0)

        query = fake_munsit.state.listen_received_query
        assert query["encoding"] == "linear16"
        assert query["sample_rate"] == "16000"
        assert query["endpointing"] == "300"
        assert query["smart_turn"] == "false"
        assert query["model"] == "munsit"
        assert query["hotwords"] == "عبد القادر,أديب"
        assert query["correlation_id"] == "call-1"
        # audio arrives as raw binary, not JSON
        assert fake_munsit.state.listen_binary_frames >= 1
        assert fake_munsit.state.listen_binary_bytes >= 2 * 1600  # >= one 100ms 16k frame

    async def test_en_ar_default_sends_model_and_drops_hotwords(self, fake_munsit, http_session):
        # Default model is munsit-en-ar: the socket query must carry it (the server
        # routes to the code-switch engine on it), hotwords must NOT be sent (no
        # biasing on that backend), and the batch "ar-en" language spelling must be
        # normalized to the live protocol's "ar" instead of failing with 4002.
        with pytest.warns(UserWarning, match="munsit-en-ar"):
            stt_inst = STT(
                mode="streaming",
                api_key="k",
                base_url=fake_munsit.url,
                http_session=http_session,
                language="ar-en",
                hotwords=["term"],
            )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        await _drive_listen_stream(stream, [_silence_frame()], q, max_wait=3.0)

        query = fake_munsit.state.listen_received_query
        assert query["model"] == "munsit-en-ar"
        assert "hotwords" not in query
        assert query["language"] == "ar"

    async def test_close_stream_sent_on_end_input(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        await _drive_listen_stream(stream, [_silence_frame()], q, max_wait=3.0)
        ctypes = [c.get("type") for c in fake_munsit.state.listen_controls]
        assert "CloseStream" in ctypes

    async def test_resamples_48k_input(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        n_frames = 5
        await _drive_listen_stream(
            stream,
            [_silence_frame(duration_ms=100, sample_rate=48000) for _ in range(n_frames)],
            q,
            max_wait=3.0,
        )
        sent = fake_munsit.state.listen_binary_bytes
        expected = n_frames * 1600 * 2  # 100ms at 16k, 16-bit
        # resampler filter delay loses a little; must be well below 48k passthrough
        assert 0.5 * expected <= sent <= 1.2 * expected

    async def test_header_auth(self, fake_munsit, http_session):
        fake_munsit.state.require_api_key = "secret"
        stt_inst = STT(
            mode="streaming",
            api_key="secret",
            base_url=fake_munsit.url,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        await _drive_listen_stream(stream, [_silence_frame()], q, max_wait=3.0)
        assert fake_munsit.state.listen_received_headers.get("x-api-key") == "secret"

    async def test_query_auth_uses_api_key_param(self, fake_munsit, http_session):
        fake_munsit.state.require_api_key = "qk"
        stt_inst = STT(
            mode="streaming",
            api_key="qk",
            base_url=fake_munsit.url,
            http_session=http_session,
            auth_method="query",
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        await _drive_listen_stream(stream, [_silence_frame()], q, max_wait=3.0)
        assert fake_munsit.state.listen_received_query.get("api_key") == "qk"

    async def test_auth_rejection_raises_status_error(self, fake_munsit, http_session):
        from livekit.agents import APIStatusError

        fake_munsit.state.require_api_key = "right"
        stt_inst = STT(
            mode="streaming",
            api_key="wrong",
            base_url=fake_munsit.url,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        with pytest.raises(APIStatusError):
            stream.push_frame(_silence_frame())
            stream.end_input()
            async for _ in stream:
                pass
        await stream.aclose()

    async def test_reconnects_after_abrupt_server_close(self, fake_munsit, http_session):
        fake_munsit.state.close_after_bytes = 1600  # close mid-stream after ~50ms of audio
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream()
        q: asyncio.Queue = asyncio.Queue()
        collector = asyncio.create_task(_collect_events_to_queue(stream, q))
        try:
            for _ in range(3):
                stream.push_frame(_silence_frame())
                await asyncio.sleep(0.05)
            fake_munsit.state.close_after_bytes = None  # let the reconnect survive
            for _ in range(60):
                await asyncio.sleep(0.05)
                if fake_munsit.state.listen_controls or fake_munsit.state.listen_binary_frames:
                    pass
                # reconnected when a fresh query dict was recorded after the close
            await asyncio.sleep(1.5)  # cover the 1s backoff
            stream.push_frame(_silence_frame())
            await asyncio.sleep(0.3)
        finally:
            collector.cancel()
            await stream.aclose()
        # server saw at least one frame after the reconnect
        assert fake_munsit.state.listen_binary_frames >= 2


# ---------------------------------------------------------------------------
# Batch (HTTP) mode tests — exercise the /api/v1/audio/transcribe path.
# ---------------------------------------------------------------------------
class TestBatchMode:
    """Batch mode buffers audio per utterance and POSTs to the HTTP endpoint
    on flush. The fake server's _batch_handler returns a canned response with
    word-level timestamps; tests verify the plugin parses them correctly."""

    async def test_emits_final_with_word_timestamps(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="batch",
            api_key="x",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))

        events: list = []
        q: asyncio.Queue = asyncio.Queue()

        async def collector(s: object) -> None:
            try:
                async for ev in s:  # type: ignore[attr-defined]
                    await q.put(ev)
            except Exception:
                pass

        col = asyncio.create_task(collector(stream))
        try:
            for _ in range(5):
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.01)
            stream.end_input()
            for _ in range(60):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                    events.append(ev)
                    if ev.type == SpeechEventType.END_OF_SPEECH:
                        break
                except asyncio.TimeoutError:
                    continue
        finally:
            col.cancel()
            await stream.aclose()

        types = [e.type for e in events]
        assert SpeechEventType.START_OF_SPEECH in types
        assert SpeechEventType.FINAL_TRANSCRIPT in types
        assert SpeechEventType.END_OF_SPEECH in types

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        sd = finals[0].alternatives[0]
        assert sd.text == "أبي أحول درهم لفرحان"
        # Word timestamps from the fake response should populate SpeechData.words.
        # TimedString is a str subclass, so equality with a plain str works.
        assert len(sd.words) == 4
        assert sd.words[0] == "أبي"
        assert sd.words[0].start_time == 0.08
        assert sd.words[0].end_time == 0.2
        assert sd.words[3] == "لفرحان"

    async def test_multipart_upload_shape(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="batch",
            api_key="my-test-key",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
            model="munsit-en-ar",
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        try:
            for _ in range(3):
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.01)
            stream.end_input()
            await asyncio.sleep(1.0)  # let the POST complete
        finally:
            await stream.aclose()

        assert fake_munsit.state.batch_received_count == 1
        assert fake_munsit.state.batch_received_models == ["munsit-en-ar"]
        # Audio should have been uploaded (header + 3 × 100ms × 16kHz × 2 bytes = 44 + 9600).
        assert fake_munsit.state.batch_received_audio_sizes[0] >= 44 + (16000 // 10 * 2 * 3)
        assert fake_munsit.state.batch_received_headers.get("x-api-key") == "my-test-key"

    async def test_bearer_auth(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="batch",
            api_key="bearer-key",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
            auth_method="bearer",
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        try:
            stream.push_frame(_silence_frame(duration_ms=100))
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(1.0)
        finally:
            await stream.aclose()

        assert fake_munsit.state.batch_received_headers.get("Authorization") == "Bearer bearer-key"
        assert "x-api-key" not in fake_munsit.state.batch_received_headers

    async def test_query_auth(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="batch",
            api_key="query-key",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
            auth_method="query",
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        try:
            stream.push_frame(_silence_frame(duration_ms=100))
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(1.0)
        finally:
            await stream.aclose()

        assert fake_munsit.state.batch_received_query.get("token") == "query-key"
        assert "x-api-key" not in fake_munsit.state.batch_received_headers

    async def test_server_error_raises_status_error(self, fake_munsit, http_session):
        from livekit.agents import APIStatusError

        fake_munsit.state.batch_response_status = 500
        fake_munsit.state.batch_response_body = {"message": "internal error"}
        stt_inst = STT(
            mode="batch",
            api_key="x",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))

        with pytest.raises(APIStatusError) as exc_info:
            try:
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.05)
                stream.end_input()
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()
        assert exc_info.value.status_code == 500

    async def test_recognize_sync_batch(self, fake_munsit, http_session):
        """STT.recognize() should call the HTTP batch endpoint synchronously."""
        stt_inst = STT(
            mode="batch",
            api_key="x",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
        )
        # Build a buffer of a few silent frames.
        frames = [_silence_frame(duration_ms=100) for _ in range(5)]
        event = await stt_inst.recognize(frames, conn_options=APIConnectOptions(max_retry=0))

        assert event.type == SpeechEventType.FINAL_TRANSCRIPT
        sd = event.alternatives[0]
        assert sd.text == "أبي أحول درهم لفرحان"
        assert len(sd.words) == 4

    async def test_speech_then_silence_triggers_submit(self, fake_munsit, http_session):
        """Mimics the AgentSession flow: VAD-detected speech, then silence
        frames pushed by AgentSession after end-of-speech. Batch mode's
        internal energy filter should detect the silence and submit
        *without* needing an explicit flush."""
        stt_inst = STT(
            mode="batch",
            api_key="x",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))

        events: list = []
        q: asyncio.Queue = asyncio.Queue()

        async def collector(s: object) -> None:
            try:
                async for ev in s:  # type: ignore[attr-defined]
                    await q.put(ev)
            except Exception:
                pass

        col = asyncio.create_task(collector(stream))
        try:
            # Loud frames simulate user speech
            for _ in range(5):
                stream.push_frame(_loud_frame(duration_ms=100))
                await asyncio.sleep(0.01)
            # Silence frames simulate AgentSession's post-EOU silence push
            for _ in range(10):
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.01)
            # Note: NO flush, NO end_input — relies on internal VAD.
            for _ in range(80):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                    events.append(ev)
                    if ev.type == SpeechEventType.END_OF_SPEECH:
                        break
                except asyncio.TimeoutError:
                    continue
        finally:
            col.cancel()
            await stream.aclose()

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) >= 1, (
            f"VAD-driven submit didn't fire; events: {[e.type for e in events]}"
        )
        assert fake_munsit.state.batch_received_count >= 1

    async def test_multiple_utterances_per_stream(self, fake_munsit, http_session):
        """A single stream can be reused for multiple utterances; each flush submits a batch."""
        stt_inst = STT(
            mode="batch",
            api_key="x",
            batch_base_url=fake_munsit.batch_url,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        events: list = []
        q: asyncio.Queue = asyncio.Queue()

        async def collector(s: object) -> None:
            try:
                async for ev in s:  # type: ignore[attr-defined]
                    await q.put(ev)
            except Exception:
                pass

        col = asyncio.create_task(collector(stream))
        try:
            # First utterance
            for _ in range(3):
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.01)
            stream.flush()
            await asyncio.sleep(0.5)
            # Second utterance
            for _ in range(3):
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.01)
            stream.end_input()
            for _ in range(60):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                    events.append(ev)
                except asyncio.TimeoutError:
                    if len([e for e in events if e.type == SpeechEventType.END_OF_SPEECH]) >= 2:
                        break
        finally:
            col.cancel()
            await stream.aclose()

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 2
        assert fake_munsit.state.batch_received_count == 2


# ---------------------------------------------------------------------------
# Integration tests — hit the real Munsit API
# ---------------------------------------------------------------------------
# sample_arabic.wav is a synthetic 440 Hz tone (2 s, 16 kHz, mono 16-bit).
# It exercises the full network path.  Replace with a genuine Arabic recording
# once one is available to also validate transcript correctness.
@pytest.mark.integration
class TestMunsitIntegration:
    @pytest.mark.skipif(
        not os.environ.get("MUNSIT_API_KEY"),
        reason="MUNSIT_API_KEY not set; skipping live integration test",
    )
    async def test_real_munsit_transcribes_sample(self, http_session):
        wav_path = pathlib.Path(__file__).parent / "sample_arabic.wav"
        assert wav_path.exists(), "sample_arabic.wav missing — run Task 19 Step 2"

        stt_inst = STT(mode="streaming", http_session=http_session)  # real key from env
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        events: list = []
        q: asyncio.Queue = asyncio.Queue()

        async def _collector(s: object) -> None:
            try:
                async for ev in s:  # type: ignore[attr-defined]
                    await q.put(ev)
            except Exception:
                pass

        collector = asyncio.create_task(_collector(stream))
        try:
            with wave.open(str(wav_path), "rb") as wf:
                sr = wf.getframerate()
                samples_per_chunk = sr // 10  # 100ms
                while True:
                    raw = wf.readframes(samples_per_chunk)
                    if not raw:
                        break
                    frame = rtc.AudioFrame(
                        data=raw,
                        sample_rate=sr,
                        num_channels=1,
                        samples_per_channel=len(raw) // 2,
                    )
                    stream.push_frame(frame)
                    await asyncio.sleep(0.1)
            stream.flush()
            for _ in range(50):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                events.append(ev)
                if ev.type == SpeechEventType.END_OF_SPEECH:
                    break
        finally:
            collector.cancel()
            await stream.aclose()

        types = [e.type for e in events]
        assert SpeechEventType.START_OF_SPEECH in types
        assert SpeechEventType.FINAL_TRANSCRIPT in types

    @pytest.mark.skipif(
        not os.environ.get("MUNSIT_API_KEY"),
        reason="MUNSIT_API_KEY not set; skipping live integration test",
    )
    async def test_real_munsit_batch_transcribe(self, http_session):
        """Hit the real /api/v1/audio/transcribe endpoint and verify word timestamps."""
        wav_path = pathlib.Path(__file__).parent / "sample_arabic.wav"
        assert wav_path.exists()

        stt_inst = STT(mode="batch", http_session=http_session)
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            data = wf.readframes(wf.getnframes())
            frame = rtc.AudioFrame(
                data=data,
                sample_rate=sr,
                num_channels=1,
                samples_per_channel=len(data) // 2,
            )

        event = await stt_inst.recognize([frame], conn_options=APIConnectOptions(max_retry=0))
        assert event.type == SpeechEventType.FINAL_TRANSCRIPT
        sd = event.alternatives[0]
        assert sd.text, "expected non-empty transcript from real Munsit batch endpoint"
        assert sd.end_time > 0


class TestTTSConsolidation:
    """The consolidated package exposes the (former livekit-plugins-faseeh) TTS."""

    def test_tts_importable_and_exported(self):
        from livekit.plugins.munsit import DEFAULT_VOICE_ID, TTS, Voice, VoiceSettings  # noqa: F401

        assert DEFAULT_VOICE_ID

    def test_tts_reads_munsit_env_var(self):
        from livekit.plugins.munsit import TTS

        with patch.dict(os.environ, {"MUNSIT_API_KEY": "env-key"}, clear=True):
            tts = TTS()
            assert tts is not None

    def test_tts_accepts_legacy_faseeh_env_var(self):
        from livekit.plugins.munsit import TTS

        with patch.dict(os.environ, {"FASEEH_API_KEY": "legacy-key"}, clear=True):
            tts = TTS()
            assert tts is not None

    def test_tts_requires_some_key(self):
        from livekit.plugins.munsit import TTS

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                TTS()
