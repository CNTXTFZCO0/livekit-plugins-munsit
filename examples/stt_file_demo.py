"""Stream a local WAV file through Munsit STT and print every event with metrics.

This example does NOT need a LiveKit room or microphone — it's the fastest sanity check
for the plugin. Run it from a checkout of livekit/agents:

    cd livekit-plugins/livekit-plugins-munsit/examples
    cp .env.example .env  # then edit .env to set MUNSIT_API_KEY
    uv run python stt_file_demo.py path/to/sample.wav
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import sys
import time
import wave
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins import munsit

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stt_file_demo")


class Metrics:
    def __init__(self) -> None:
        self.audio_started_at: float | None = None
        self.first_interim_at: float | None = None
        self.last_interim_at: float | None = None
        self.interim_count = 0
        self.interim_gaps: list[float] = []
        self.final_at: float | None = None
        self.final_text: str = ""
        self.audio_seconds_sent: float = 0.0


async def stream_wav(path: Path, metrics: Metrics, model: str = "munsit") -> None:
    # Pass an explicit aiohttp session so this script works outside a LiveKit job context.
    async with aiohttp.ClientSession() as http_session:
        await _run_stream(path, metrics, http_session, model)


async def _run_stream(
    path: Path,
    metrics: Metrics,
    http_session: aiohttp.ClientSession,
    model: str = "munsit",
) -> None:
    stt_inst = munsit.STT(model=model, http_session=http_session)  # MUNSIT_API_KEY from env
    stream = stt_inst.stream()
    metrics.audio_started_at = time.monotonic()

    async def reader() -> None:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                raise SystemExit(f"only 16-bit PCM WAV supported (got {sampwidth * 8}-bit)")
            chunk_ms = 100
            samples_per_chunk = sr * chunk_ms // 1000
            while True:
                raw = wf.readframes(samples_per_chunk)
                if not raw:
                    break
                frame = rtc.AudioFrame(
                    data=raw,
                    sample_rate=sr,
                    num_channels=nch,
                    samples_per_channel=len(raw) // (nch * sampwidth),
                )
                stream.push_frame(frame)
                metrics.audio_seconds_sent += chunk_ms / 1000.0
                # Pace at 1× real-time so Munsit can keep up.
                await asyncio.sleep(chunk_ms / 1000.0)
        # Signal end of input WITHOUT cancelling the main task. The plugin's
        # internal drain logic then waits for the server to flush remaining
        # transcripts before tearing down.
        #
        # Do NOT use stream.aclose() here — it cancels the main task immediately
        # and kills any in-flight drain, producing empty transcripts when the
        # server is slow to respond.
        stream.end_input()

    async def event_logger() -> None:
        try:
            async for ev in stream:
                _print_event(ev, metrics)
        except Exception as e:
            logger.error("stream raised: %s", e)

    try:
        await asyncio.gather(reader(), event_logger(), return_exceptions=False)
    finally:
        # Final cleanup; main task has already finished naturally via end_input.
        await stream.aclose()


def _print_event(ev: SpeechEvent, m: Metrics) -> None:
    base = m.audio_started_at or time.monotonic()
    rel_ms = round((time.monotonic() - base) * 1000)
    if ev.type == SpeechEventType.START_OF_SPEECH:
        print(f"[+{rel_ms}ms] START_OF_SPEECH      req={ev.request_id}")
    elif ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
        text = ev.alternatives[0].text if ev.alternatives else ""
        now = time.monotonic()
        if m.first_interim_at is None:
            m.first_interim_at = now
        if m.last_interim_at is not None:
            m.interim_gaps.append((now - m.last_interim_at) * 1000)
        m.last_interim_at = now
        m.interim_count += 1
        print(f'[+{rel_ms}ms] INTERIM_TRANSCRIPT   "{text}"')
    elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
        sd = ev.alternatives[0] if ev.alternatives else None
        text = sd.text if sd else ""
        m.final_at = time.monotonic()
        m.final_text = text
        print(f'[+{rel_ms}ms] FINAL_TRANSCRIPT     "{text}"')
        # Surface word timestamps when batch mode returns them.
        if sd and sd.words:
            print("           word timings:")
            for w in sd.words:
                print(
                    f"             {float(w.start_time):5.2f}s → {float(w.end_time):5.2f}s  {w!s}"
                )
    elif ev.type == SpeechEventType.END_OF_SPEECH:
        print(f"[+{rel_ms}ms] END_OF_SPEECH")
        _print_metrics_summary(m)
        m.first_interim_at = None
        m.last_interim_at = None
        m.interim_count = 0
        m.interim_gaps = []
        m.final_at = None
        m.final_text = ""
    elif ev.type == SpeechEventType.RECOGNITION_USAGE and ev.recognition_usage:
        print(f"[+{rel_ms}ms] RECOGNITION_USAGE    {ev.recognition_usage.audio_duration:.2f}s")


def _print_metrics_summary(m: Metrics) -> None:
    if m.audio_started_at is None:
        return
    parts = []
    if m.first_interim_at is not None:
        parts.append(f"ttfi={round((m.first_interim_at - m.audio_started_at) * 1000)} ms")
    if m.final_at is not None:
        parts.append(f"ttf={round((m.final_at - m.audio_started_at) * 1000)} ms")
    parts.append(f"interim_updates={m.interim_count}")
    if m.interim_gaps:
        p50 = round(statistics.median(m.interim_gaps))
        parts.append(f"interim_p50_gap={p50} ms")
    parts.append(f"audio_sent={m.audio_seconds_sent:.2f} s")
    print("Utterance metrics: " + ", ".join(parts))


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: stt_file_demo.py <path-to-wav> [model]\n"
            "  model: 'munsit' (default, Arabic) or 'munsit-en-ar' (Arabic+English)\n"
            "         can also be set via MUNSIT_MODEL env var",
            file=sys.stderr,
        )
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        sys.exit(2)
    import os

    model = sys.argv[2] if len(sys.argv) >= 3 else os.environ.get("MUNSIT_MODEL", "munsit")
    print(f"# using model: {model}", file=sys.stderr)
    metrics = Metrics()
    asyncio.run(stream_wav(path, metrics, model=model))


if __name__ == "__main__":
    main()
