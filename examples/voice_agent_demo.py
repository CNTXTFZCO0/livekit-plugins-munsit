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
from livekit.agents import Agent, AgentServer, AgentSession, ChatMessage
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
    """Per-turn latency rollup using the conversation_item_added event.

    LiveKit deprecated the ``metrics_collected`` event; per-turn metrics now
    live on ``ChatMessage.metrics`` (a ``MetricsReport`` TypedDict). User
    messages carry ``transcription_delay`` and ``end_of_turn_delay``;
    assistant messages carry ``llm_node_ttft`` and ``tts_node_ttfb``.

    We accumulate the user-side metrics when a user message lands, then
    print the full rollup when the matching assistant message lands.
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.transcription_delay_ms: float | None = None
        self.eou_ms: float | None = None
        self.llm_ttft_ms: float | None = None
        self.tts_ttfb_ms: float | None = None

    def on_chat_message(self, msg: ChatMessage) -> None:
        m = msg.metrics or {}
        if msg.role == "user":
            td = m.get("transcription_delay")
            eou = m.get("end_of_turn_delay")
            if td is not None:
                self.transcription_delay_ms = round(td * 1000)
            if eou is not None:
                self.eou_ms = round(eou * 1000)
        elif msg.role == "assistant":
            llm_ttft = m.get("llm_node_ttft")
            tts_ttfb = m.get("tts_node_ttfb")
            if llm_ttft is not None and llm_ttft > 0:
                self.llm_ttft_ms = round(llm_ttft * 1000)
            if tts_ttfb is not None and tts_ttfb > 0:
                self.tts_ttfb_ms = round(tts_ttfb * 1000)
            self._print_summary()

    def _print_summary(self) -> None:
        parts: list[str] = []
        total = 0.0
        if self.eou_ms is not None:
            parts.append(f"EOU {self.eou_ms} ms")
            total += self.eou_ms
        if self.transcription_delay_ms is not None:
            parts.append(f"STT {self.transcription_delay_ms} ms")
            total += self.transcription_delay_ms
        if self.llm_ttft_ms is not None:
            parts.append(f"LLM {self.llm_ttft_ms} ms")
            total += self.llm_ttft_ms
        if self.tts_ttfb_ms is not None:
            parts.append(f"TTS {self.tts_ttfb_ms} ms")
            total += self.tts_ttfb_ms
        if not parts:
            return  # nothing to print yet
        logger.info("Turn latency: %s = %.2fs", " + ".join(parts), total / 1000)
        self._reset()


_tracker = TurnTracker()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=munsit.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        # Use Silero VAD for interruption detection. LiveKit's hosted
        # adaptive detector at agent-gateway.livekit.cloud needs LiveKit
        # Cloud auth and isn't necessary for a Munsit-based agent demo.
        turn_handling={"interruption": {"mode": "vad"}},
    )

    @session.on("conversation_item_added")
    def _on_item(event):  # type: ignore[no-untyped-def]
        if isinstance(event.item, ChatMessage):
            _tracker.on_chat_message(event.item)

    await session.start(room=ctx.room, agent=BilingualAssistant())
    await session.generate_reply(instructions="Greet the user warmly in Arabic.")


if __name__ == "__main__":
    agents.cli.run_app(server)
