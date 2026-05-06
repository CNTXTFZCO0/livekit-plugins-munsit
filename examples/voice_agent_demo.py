"""Full bilingual voice agent: Munsit STT + OpenAI LLM + Cartesia TTS + Silero VAD.

Mirrors the faseeh ``simple_agent.py`` so users can compare and contrast.

Run with::

    cd livekit-plugins/livekit-plugins-munsit/examples
    cp .env.example .env  # set MUNSIT_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY, LIVEKIT_*
    uv run python voice_agent_demo.py dev
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.agents.metrics import (
    EOUMetrics,
    LLMMetrics,
    RealtimeModelMetrics,
    STTMetrics,
    TTSMetrics,
)
from livekit.plugins import cartesia, munsit, openai, silero

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("voice_agent_demo")

server = AgentServer()


class BilingualAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly bilingual assistant. Reply in the language the user speaks. "
                "Keep responses brief."
            )
        )


class TurnTracker:
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.eou_ms: float | None = None
        self.stt_audio_s: float | None = None
        self.llm_ttft_ms: float | None = None
        self.tts_ttfb_ms: float | None = None

    def record(self, m: object) -> None:
        if isinstance(m, EOUMetrics):
            self.eou_ms = round(m.end_of_utterance_delay * 1000)
        elif isinstance(m, STTMetrics):
            self.stt_audio_s = round(m.audio_duration, 2)
        elif isinstance(m, (RealtimeModelMetrics, LLMMetrics)):
            ttft = getattr(m, "ttft", None)
            if ttft is not None and ttft > 0:
                self.llm_ttft_ms = round(ttft * 1000)
        elif isinstance(m, TTSMetrics):
            self.tts_ttfb_ms = round(m.ttfb * 1000)
            self._print_summary()

    def _print_summary(self) -> None:
        parts: list[str] = []
        total = 0.0
        if self.eou_ms is not None:
            parts.append(f"EOU {self.eou_ms} ms")
            total += self.eou_ms
        if self.llm_ttft_ms is not None:
            parts.append(f"LLM {self.llm_ttft_ms} ms")
            total += self.llm_ttft_ms
        if self.tts_ttfb_ms is not None:
            parts.append(f"TTS {self.tts_ttfb_ms} ms")
            total += self.tts_ttfb_ms
        audio_info = f" | stt_audio={self.stt_audio_s}s" if self.stt_audio_s is not None else ""
        logger.info(
            "Turn latency: %s = %.2fs%s",
            " + ".join(parts),
            total / 1000,
            audio_info,
        )
        self._reset()


_tracker = TurnTracker()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=munsit.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    @session.on("metrics_collected")
    def _on_metrics(event):  # type: ignore[no-untyped-def]
        _tracker.record(event.metrics)

    await session.start(room=ctx.room, agent=BilingualAssistant())
    await session.generate_reply(instructions="Greet the user warmly in Arabic.")


if __name__ == "__main__":
    agents.cli.run_app(server)
