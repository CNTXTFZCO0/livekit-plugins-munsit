# Copyright 2026 LiveKit, Inc.
import asyncio
import json as _json
import os
import pathlib
import struct
import wave
from unittest.mock import patch

import aiohttp
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
        assert stt.model == "munsit"
        assert stt.provider == "Munsit"

    def test_custom_model(self):
        stt = STT(api_key="x", model="munsit-en-ar")
        assert stt.model == "munsit-en-ar"

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

    def test_invalid_finalize_threshold(self):
        with pytest.raises(ValueError, match="finalize_after_silence_ms"):
            STT(api_key="x", finalize_after_silence_ms=50)

    def test_invalid_vad_silence(self):
        with pytest.raises(ValueError, match="vad_silence_ms"):
            STT(api_key="x", vad_silence_ms=50)

    def test_default_mode_is_batch(self):
        stt = STT(api_key="x")
        assert stt._opts.mode == "batch"

    def test_capabilities_batch_default(self):
        stt = STT(api_key="x")
        assert stt.capabilities.streaming is True
        # batch mode never emits interims regardless of the flag
        assert stt.capabilities.interim_results is False
        # batch returns word timestamps from the HTTP response
        assert stt.capabilities.aligned_transcript == "word"

    def test_capabilities_streaming(self):
        stt = STT(api_key="x", mode="streaming")
        assert stt.capabilities.streaming is True
        assert stt.capabilities.interim_results is True
        assert stt.capabilities.aligned_transcript is False

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

    def test_update_options_invalid_finalize_threshold(self):
        stt = STT(api_key="x")
        with pytest.raises(ValueError, match="finalize_after_silence_ms"):
            stt.update_options(finalize_after_silence_ms=50)

    def test_update_options_invalid_vad_silence(self):
        stt = STT(api_key="x")
        with pytest.raises(ValueError, match="vad_silence_ms"):
            stt.update_options(vad_silence_ms=50)


class TestFakeMunsitServer:
    async def test_server_accepts_ws_connection(self, fake_munsit):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(fake_munsit.url) as ws:
                await ws.send_str('{"event":"audio_chunk","data":{"audioBuffer":[1,2,3]}}')
                # Server has no script; close cleanly.
                await ws.close()
        assert len(fake_munsit.state.received_messages) == 1
        assert fake_munsit.state.received_messages[0]["event"] == "audio_chunk"

    async def test_server_replays_script(self, fake_munsit):
        fake_munsit.script(
            [
                (0.0, "transcription", "مر"),
                (0.05, "transcription", "مرحبا"),
            ]
        )
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(fake_munsit.url) as ws:
                received = []
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        received.append(msg.data)
                    if len(received) == 2:
                        await ws.close()
                        break
        assert len(received) == 2
        assert _json.loads(received[0])["data"] == "مر"
        assert _json.loads(received[1])["data"] == "مرحبا"


def _silence_frame(duration_ms: int = 100, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * duration_ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class TestSpeechStreamHappyPath:
    async def test_first_chunk_includes_wav_header(self, fake_munsit, http_session):
        fake_munsit.script([(0.05, "transcription", "test")])
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream()
        try:
            stream.push_frame(_silence_frame())
            # Wait briefly for the WS round-trip and the scripted transcription.
            async for ev in stream:
                if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    break
        finally:
            await stream.aclose()

        # First message MUST contain a WAV header at the start of audioBuffer.
        first = fake_munsit.state.received_messages[0]
        assert first["event"] == "audio_chunk"
        buf = first["data"]["audioBuffer"]
        assert buf[0:4] == [ord("R"), ord("I"), ord("F"), ord("F")]
        assert buf[8:12] == [ord("W"), ord("A"), ord("V"), ord("E")]

    async def test_subsequent_chunks_no_wav_header(self, fake_munsit, http_session):
        fake_munsit.script([(0.5, "transcription", "x")])
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        stream = stt_inst.stream()
        try:
            for _ in range(3):
                stream.push_frame(_silence_frame())
            await asyncio.sleep(0.2)
        finally:
            await stream.aclose()

        # At least 2 audio_chunk messages — the second must not start with RIFF.
        chunks = [m for m in fake_munsit.state.received_messages if m["event"] == "audio_chunk"]
        assert len(chunks) >= 2
        second_buf = chunks[1]["data"]["audioBuffer"]
        assert second_buf[0:4] != [ord("R"), ord("I"), ord("F"), ord("F")]

    async def test_emits_interim_for_cumulative_update(self, fake_munsit, http_session):
        fake_munsit.script(
            [
                (0.05, "transcription", "مر"),
                (0.15, "transcription", "مرحبا"),
            ]
        )
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            http_session=http_session,
            finalize_after_silence_ms=10000,
        )
        stream = stt_inst.stream()
        events: list = []
        try:
            stream.push_frame(_silence_frame())
            async for ev in stream:
                events.append(ev)
                is_target_interim = (
                    ev.type == SpeechEventType.INTERIM_TRANSCRIPT
                    and ev.alternatives[0].text == "مرحبا"
                )
                if is_target_interim:
                    break
        finally:
            await stream.aclose()

        interims = [e for e in events if e.type == SpeechEventType.INTERIM_TRANSCRIPT]
        assert any(e.alternatives[0].text == "مر" for e in interims)
        assert any(e.alternatives[0].text == "مرحبا" for e in interims)


async def _collect_events_to_queue(stream: object, q: asyncio.Queue) -> None:  # type: ignore[type-arg]
    """Background helper: drain stream events into a queue so tests can poll with timeouts
    without cancelling the underlying async generator (which would corrupt the tee_peer state)."""
    async for ev in stream:  # type: ignore[union-attr]
        await q.put(ev)


class TestServerErrors:
    async def test_transcription_error_raises_status_error(self, fake_munsit, http_session):
        from livekit.agents import APIConnectOptions, APIStatusError

        fake_munsit.script([(0.05, "transcription_error", "auth quota exceeded")])
        stt_inst = STT(
            mode="streaming", api_key="x", base_url=fake_munsit.url, http_session=http_session
        )
        # max_retry=0 so the APIStatusError propagates directly without being wrapped
        # in APIConnectionError after exhausted retries.
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        with pytest.raises(APIStatusError) as exc_info:
            try:
                stream.push_frame(_silence_frame())
                async for _ev in stream:
                    pass
            finally:
                await stream.aclose()
        assert exc_info.value.status_code == 500
        assert "auth quota exceeded" in exc_info.value.message


class TestServerDiffFinalization:
    async def test_emits_final_after_idle(self, fake_munsit, http_session):
        from livekit.agents import APIConnectOptions

        fake_munsit.script(
            [
                (0.0, "transcription", "مر"),
                (0.05, "transcription", "مرحبا"),
                # No more updates — idle timer should fire.
            ]
        )
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            http_session=http_session,
            finalize_after_silence_ms=200,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        collector = asyncio.create_task(_collect_events_to_queue(stream, q))
        events: list = []
        try:
            stream.push_frame(_silence_frame())
            for _ in range(50):  # up to 5s budget
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
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
        assert SpeechEventType.INTERIM_TRANSCRIPT in types
        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == "مرحبا"
        assert types[-1] == SpeechEventType.END_OF_SPEECH

    async def test_resets_for_second_utterance(self, fake_munsit, http_session):
        from livekit.agents import APIConnectOptions

        fake_munsit.script(
            [
                (0.0, "transcription", "أ"),
                (0.05, "transcription", "أهلا"),
                # 0.5s gap → first idle fires at ~0.25s, then silence; second batch starts at ~0.55s
                (0.5, "transcription", "أهلا كيف"),
                (0.05, "transcription", "أهلا كيف حالك"),
            ]
        )
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            http_session=http_session,
            finalize_after_silence_ms=200,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        q: asyncio.Queue = asyncio.Queue()
        collector = asyncio.create_task(_collect_events_to_queue(stream, q))
        events: list = []
        try:
            stream.push_frame(_silence_frame())
            for _ in range(80):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                events.append(ev)
                if (
                    ev.type == SpeechEventType.END_OF_SPEECH
                    and len([e for e in events if e.type == SpeechEventType.END_OF_SPEECH]) == 2
                ):
                    break
        finally:
            collector.cancel()
            await stream.aclose()

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 2
        assert finals[0].alternatives[0].text == "أهلا"
        assert finals[1].alternatives[0].text == "أهلا كيف حالك"


class TestDrainOnClose:
    """The plugin must not lose FINAL_TRANSCRIPT events when the consumer
    closes the stream before the server has time to emit its first response.

    Reproduces the bug we observed in real Munsit testing where ~half of
    runs returned no FINAL_TRANSCRIPT at all because the post-audio wait
    in user code was shorter than Munsit's first-response latency.

    Important: the consumer must call ``end_input()`` (not ``aclose()``)
    after the last frame to give the drain a chance to run. ``aclose()``
    cancels the main task immediately and kills any in-flight drain.
    """

    async def test_drain_waits_for_late_first_cumulative(self, fake_munsit, http_session):
        # Server emits its first (and only) transcription 1.5 seconds after
        # the WS opens. A naive close-on-input-drain would tear down the WS
        # before this arrives.
        fake_munsit.script([(1.5, "transcription", "متأخر")])
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            http_session=http_session,
            finalize_after_silence_ms=300,
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
            # Push a brief audio (< 1 s of frames) and signal end of input
            # immediately. The cumulative arrives during drain.
            for _ in range(5):
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.02)
            stream.end_input()  # closes input_ch without cancelling main task
            # Drain in the plugin should now have time to receive "متأخر"
            # and emit FINAL via the idle timer.
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

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) >= 1, f"no FINAL emitted; events: {[e.type for e in events]}"
        assert finals[0].alternatives[0].text == "متأخر"


class TestFlushSentinel:
    async def test_flush_finalizes_immediately(self, fake_munsit, http_session):
        import time

        from livekit.agents import APIConnectOptions

        fake_munsit.script([(0.0, "transcription", "نص")])
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            finalize_after_silence_ms=10000,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))

        events: list = []
        q: asyncio.Queue = asyncio.Queue()

        async def _collector(s: object) -> None:
            async for ev in s:  # type: ignore[attr-defined]
                await q.put(ev)

        collector = asyncio.create_task(_collector(stream))
        try:
            stream.push_frame(_silence_frame())
            # Wait for first INTERIM
            for _ in range(20):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                events.append(ev)
                if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    break

            # Now flush — expect FINAL within ~1s instead of 10s.
            t0 = time.monotonic()
            stream.flush()
            for _ in range(20):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                events.append(ev)
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    break
            elapsed = time.monotonic() - t0
            assert elapsed < 2.0, f"flush did not finalize promptly (took {elapsed:.2f}s)"
        finally:
            collector.cancel()
            await stream.aclose()

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == "نص"


class TestAuthMethods:
    async def _connect_and_close(self, fake_munsit, http_session, **kwargs) -> None:
        from livekit.agents import APIConnectOptions

        stt_inst = STT(
            mode="streaming", base_url=fake_munsit.url, http_session=http_session, **kwargs
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))

        async def _drain(s: object) -> None:
            try:
                async for _ev in s:  # type: ignore[attr-defined]
                    pass
            except Exception:
                pass

        drain = asyncio.create_task(_drain(stream))
        try:
            stream.push_frame(_silence_frame())
            await asyncio.sleep(0.15)
        finally:
            drain.cancel()
            await stream.aclose()

    async def test_header_auth_sends_x_api_key(self, fake_munsit, http_session):
        await self._connect_and_close(
            fake_munsit, http_session, api_key="key-h", auth_method="header"
        )
        assert fake_munsit.state.received_headers.get("x-api-key") == "key-h"
        assert "Authorization" not in fake_munsit.state.received_headers
        assert "token" not in fake_munsit.state.received_query

    async def test_bearer_auth_sends_authorization(self, fake_munsit, http_session):
        await self._connect_and_close(
            fake_munsit, http_session, api_key="key-b", auth_method="bearer"
        )
        assert fake_munsit.state.received_headers.get("Authorization") == "Bearer key-b"
        assert "x-api-key" not in fake_munsit.state.received_headers

    async def test_query_auth_sends_token_param(self, fake_munsit, http_session):
        await self._connect_and_close(
            fake_munsit, http_session, api_key="key-q", auth_method="query"
        )
        assert fake_munsit.state.received_query.get("token") == "key-q"
        assert "x-api-key" not in fake_munsit.state.received_headers

    async def test_model_query_param(self, fake_munsit, http_session):
        await self._connect_and_close(
            fake_munsit,
            http_session,
            api_key="x",
            model="munsit-en-ar",
            auth_method="header",
        )
        assert fake_munsit.state.received_query.get("model") == "munsit-en-ar"

    async def test_extra_query_params(self, fake_munsit, http_session):
        await self._connect_and_close(
            fake_munsit,
            http_session,
            api_key="x",
            extra_query_params={"region": "uae", "experimental": "1"},
        )
        assert fake_munsit.state.received_query.get("region") == "uae"
        assert fake_munsit.state.received_query.get("experimental") == "1"


class TestReconnect:
    async def test_reconnects_after_server_close(self, fake_munsit, http_session):
        # Force the fake server to close the WS after receiving 1 audio_chunk per connection.
        fake_munsit.state.close_after_messages = 1
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            finalize_after_silence_ms=10000,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        # Tighten reconnect backoff so the test runs fast.
        stream._initial_backoff = 0.1
        stream._max_backoff = 0.5

        async def _drain(s: object) -> None:
            try:
                async for _ev in s:  # type: ignore[attr-defined]
                    pass
            except Exception:
                pass

        drain = asyncio.create_task(_drain(stream))
        try:
            # Push frames at real-time-ish cadence so the WS has time to receive a
            # CLOSE between sends.  Each frame triggers the server's close_after=1,
            # forcing the plugin to reconnect; the next frame goes over a fresh WS
            # (which prepends a new WAV header — that's what we count).
            for _ in range(3):
                stream.push_frame(_silence_frame())
                await asyncio.sleep(0.3)
        finally:
            drain.cancel()
            await stream.aclose()

        riff_chunks = [
            m
            for m in fake_munsit.state.received_messages
            if m.get("event") == "audio_chunk"
            and m["data"]["audioBuffer"][0:4] == [ord("R"), ord("I"), ord("F"), ord("F")]
        ]
        assert len(riff_chunks) >= 2, (
            f"expected at least 2 connections (each starts with WAV header), got {len(riff_chunks)}"
        )


class TestUpdateOptionsReconnect:
    async def test_update_model_triggers_reconnect(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            finalize_after_silence_ms=10000,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._initial_backoff = 0.1
        stream._max_backoff = 0.3

        async def _drain(s: object) -> None:
            try:
                async for _ev in s:  # type: ignore[attr-defined]
                    pass
            except Exception:
                pass

        drain = asyncio.create_task(_drain(stream))
        try:
            stream.push_frame(_silence_frame())
            await asyncio.sleep(0.5)
            stt_inst.update_options(model="munsit-en-ar")
            await asyncio.sleep(0.7)
            stream.push_frame(_silence_frame())
            await asyncio.sleep(0.5)
        finally:
            drain.cancel()
            await stream.aclose()

        # Two WS connections should have happened: one with model=munsit, one with model=munsit-en-ar.
        # Each starts with a WAV header.
        riff_chunks = [
            m
            for m in fake_munsit.state.received_messages
            if m.get("event") == "audio_chunk"
            and m["data"]["audioBuffer"][0:4] == [ord("R"), ord("I"), ord("F"), ord("F")]
        ]
        assert len(riff_chunks) >= 2, (
            f"expected at least 2 connections after update_options, got {len(riff_chunks)}"
        )
        # The most recently observed query string should reflect the new model
        # (the FakeMunsitServer overwrites received_query on each WS connect).
        assert fake_munsit.state.received_query.get("model") == "munsit-en-ar"


class TestRecognitionUsage:
    async def test_emits_recognition_usage(self, fake_munsit, http_session):
        fake_munsit.script([(0.0, "transcription", "x")])
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            finalize_after_silence_ms=10000,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))
        # Tighten the report interval so the test doesn't wait 5s.
        stream._usage_report_interval = 0.2

        events: list = []
        q: asyncio.Queue = asyncio.Queue()

        async def _collector(s: object) -> None:
            async for ev in s:  # type: ignore[attr-defined]
                await q.put(ev)

        collector = asyncio.create_task(_collector(stream))
        try:
            for _ in range(5):
                stream.push_frame(_silence_frame(duration_ms=200))
                await asyncio.sleep(0.05)
            for _ in range(20):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                events.append(ev)
                if ev.type == SpeechEventType.RECOGNITION_USAGE:
                    break
        finally:
            collector.cancel()
            await stream.aclose()

        usage = [e for e in events if e.type == SpeechEventType.RECOGNITION_USAGE]
        assert len(usage) >= 1
        assert usage[0].recognition_usage is not None
        assert usage[0].recognition_usage.audio_duration > 0


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


class TestClientVadMode:
    async def test_silence_does_not_open_ws(self, fake_munsit, http_session):
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            endpointing="client_vad",
            energy_filter=True,
            vad_silence_ms=200,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))

        async def _drain(s: object) -> None:
            try:
                async for _ev in s:  # type: ignore[attr-defined]
                    pass
            except Exception:
                pass

        drain = asyncio.create_task(_drain(stream))
        try:
            for _ in range(3):
                stream.push_frame(_silence_frame())
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.4)
        finally:
            drain.cancel()
            await stream.aclose()

        # No audio_chunk reached the server because no frames had energy.
        chunks = [m for m in fake_munsit.state.received_messages if m.get("event") == "audio_chunk"]
        assert len(chunks) == 0

    async def test_speech_then_silence_finalizes(self, fake_munsit, http_session):
        fake_munsit.script([(0.05, "transcription", "أهلا")])
        stt_inst = STT(
            mode="streaming",
            api_key="x",
            base_url=fake_munsit.url,
            endpointing="client_vad",
            energy_filter=True,
            vad_silence_ms=200,
            http_session=http_session,
        )
        stream = stt_inst.stream(conn_options=APIConnectOptions(max_retry=0))

        events: list = []
        q: asyncio.Queue = asyncio.Queue()

        async def _collector(s: object) -> None:
            async for ev in s:  # type: ignore[attr-defined]
                await q.put(ev)

        collector = asyncio.create_task(_collector(stream))
        try:
            for _ in range(5):
                stream.push_frame(_loud_frame(duration_ms=100))
                await asyncio.sleep(0.05)
            for _ in range(5):
                stream.push_frame(_silence_frame(duration_ms=100))
                await asyncio.sleep(0.05)
            for _ in range(30):
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
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
        assert SpeechEventType.END_OF_SPEECH in types


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
