"""Console STT demo: speak into the local mic, see Munsit transcripts.

Run with:

    cd livekit-plugins/livekit-plugins-munsit/examples
    cp .env.example .env  # set MUNSIT_API_KEY at minimum
    uv run python stt_console_demo.py console

To switch models, set MUNSIT_MODEL in your environment:

    MUNSIT_MODEL=munsit-en-ar uv run python stt_console_demo.py console

The script wires Munsit STT + Silero VAD into an AgentSession and
subscribes to `user_input_transcribed` events.  Every partial and final
transcript is printed to stdout so you can verify Arabic (or code-switched)
recognition without a full agent stack.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext
from livekit.plugins import munsit, silero

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("stt_console_demo")

server = AgentServer()


class _PassthroughAgent(Agent):
    """Minimal agent — no LLM or TTS needed for a pure STT demo."""

    def __init__(self) -> None:
        super().__init__(instructions="Listen and transcribe.")


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    model = os.environ.get("MUNSIT_MODEL", "munsit")
    print(f"# using model: {model}")
    session = AgentSession(
        stt=munsit.STT(model=model),
        vad=silero.VAD.load(),
        # Skip LiveKit's hosted "adaptive" interruption detector
        # (wss://agent-gateway.livekit.cloud/v1/bargein) and use Silero VAD
        # directly — avoids 401 noise when LIVEKIT_* creds aren't set up
        # for the inference gateway.
        turn_handling={"interruption": {"mode": "vad"}},
    )

    @session.on("user_input_transcribed")
    def _on_transcribed(ev: agents.UserInputTranscribedEvent) -> None:
        tag = "FINAL  " if ev.is_final else "interim"
        lang = f" [{ev.language}]" if ev.language else ""
        print(f"[{tag}]{lang} {ev.transcript}")

    await session.start(room=ctx.room, agent=_PassthroughAgent())
    print("\n--- Speak now (Arabic). Ctrl-C to exit. ---\n")


if __name__ == "__main__":
    agents.cli.run_app(server)
