---
title: LiveKit STT Plugin
description: Use Munsit speech-to-text with LiveKit Agents
---

## Munsit STT for LiveKit Agents

The `livekit-plugins-munsit` package adds Munsit speech-to-text to LiveKit Agents. It is optimized for Arabic speech recognition and supports Arabic/English code-switching through the `munsit-en-ar` model.

## Prerequisites

* A [Munsit AI](https://app.munsit.com/) account
* Python 3.10 or higher
* Basic familiarity with the LiveKit Agents framework
* A LiveKit Cloud account or self-hosted LiveKit server

## API Key

Go to [Munsit - API Keys](https://app.munsit.com//en/api-keys), generate an API key, then save it securely.

<Warning>
  Your API key is only shown once. Save it securely.
</Warning>

Set the key in your environment:

```bash theme={null}
MUNSIT_API_KEY=your_MUNSIT_API_KEY_here
```

## Installation

Install the Munsit STT plugin for LiveKit Agents:

```bash theme={null}
pip install livekit-plugins-munsit
```

## Quick Start

```python theme={null}
from livekit.agents import AgentSession
from livekit.plugins import munsit, silero

session = AgentSession(
    stt=munsit.STT(),
    vad=silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3),
    # ... llm, tts ...
)
```

`munsit.STT()` uses batch mode by default. In batch mode, LiveKit's VAD signals end-of-speech through `flush()`, then the plugin sends the buffered utterance to Munsit and emits a final transcript with word-level timestamps.

The Silero VAD thresholds above (`activation_threshold=0.6`, `min_speech_duration=0.3`) are tuned for real-world microphones, where post-echo-cancellation audio from the agent's own speaker can otherwise be misinterpreted as user speech. See [Best Practices](#best-practices) for details.

## Modes

| Mode              | When to use                                                              | Endpoint                              | Latency                                                                   | Word timestamps                      | Interim events                           |
| ----------------- | ------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------- |
| `batch` (default) | Production with `AgentSession` + VAD; transcribing recorded audio.       | `POST /api/v1/audio/transcribe`       | VAD detection + upload + server processing (\~1-2 s for short utterances) | Yes, populated on `SpeechData.words` | No                                       |
| `streaming`       | Live captions or on-the-fly UI updates while the user is still speaking. | `WS /api/v1/websocket/speech-to-text` | \~700 ms idle threshold by default                                        | No                                   | Yes, through `INTERIM_TRANSCRIPT` events |

```python theme={null}
from livekit.plugins import munsit

batch_stt = munsit.STT()
streaming_stt = munsit.STT(mode="streaming")
```

<Note>
  `STT.recognize(audio_buffer)` always uses the batch HTTP endpoint, even when `mode="streaming"` is configured.
</Note>

## Full Agent Example

Create a file called `arabic_stt_agent.py`:

```python theme={null}
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, AgentServer
from livekit.plugins import faseeh, munsit, openai, silero

load_dotenv(".env.local")


class ArabicAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""أنت مساعد صوتي ذكي يتحدث العربية بطلاقة.
            أجب على المستخدم بطريقة واضحة ومختصرة."""
        )


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=munsit.STT(
            model="munsit",
            mode="batch",
        ),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=faseeh.TTS(),
        vad=silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3),
    )

    await session.start(
        room=ctx.room,
        agent=ArabicAssistant(),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
```

Run the agent locally:

```bash theme={null}
python arabic_stt_agent.py dev
```

<Note>
  Pairing Munsit STT with [Faseeh TTS](/integrations/livekit-plugin) gives you a fully Arabic-native voice loop. If you'd rather use a non-Munsit TTS, `cartesia.TTS()` is a strong choice; `openai.TTS()` works but produces less natural Arabic prosody.
</Note>

## Model Selection

Choose the model based on the input language:

| Model          | Use case                                              |
| -------------- | ----------------------------------------------------- |
| `munsit`       | Arabic speech recognition. This is the default model. |
| `munsit-en-ar` | Mixed Arabic-English speech with code-switching.      |

```python theme={null}
stt = munsit.STT(model="munsit-en-ar")
```

## Configuration Reference

| Parameter                   | Default                         | Description                                                                |
| --------------------------- | ------------------------------- | -------------------------------------------------------------------------- |
| `mode`                      | `batch`                         | `batch` for HTTP transcription or `streaming` for WebSocket transcription. |
| `model`                     | `munsit`                        | `munsit` for Arabic or `munsit-en-ar` for Arabic/English code-switching.   |
| `api_key`                   | env `MUNSIT_API_KEY`            | Munsit API key.                                                            |
| `base_url`                  | Munsit production WebSocket URL | Override the streaming WebSocket URL.                                      |
| `batch_base_url`            | Munsit production HTTPS URL     | Override the batch HTTP URL.                                               |
| `auth_method`               | `header`                        | Authentication style: `header`, `bearer`, or `query`.                      |
| `sample_rate`               | `16000`                         | Sample rate used for the generated WAV header.                             |
| `num_channels`              | `1`                             | Number of audio channels.                                                  |
| `interim_results`           | `True`                          | Emits interim transcripts in streaming mode.                               |
| `endpointing`               | `server_diff`                   | Streaming endpointing strategy: `server_diff` or `client_vad`.             |
| `finalize_after_silence_ms` | `700`                           | Silence threshold before finalizing in `server_diff` mode.                 |
| `energy_filter`             | `False`                         | Enables energy filtering for `client_vad` mode.                            |
| `vad_silence_ms`            | `1500`                          | Silence duration used by `client_vad` mode.                                |
| `language`                  | `None`                          | Label attached to `SpeechData.language`; defaults to `ar`.                 |
| `http_session`              | `None`                          | Custom `aiohttp.ClientSession`.                                            |
| `extra_query_params`        | `None`                          | Extra query params for the streaming WebSocket endpoint.                   |

## Authentication Methods

Munsit accepts the API key in three different places. Choose whichever fits your deployment:

```python theme={null}
# Default — sends the key as the `x-api-key` header.
munsit.STT(auth_method="header")

# Authorization: Bearer <key>
munsit.STT(auth_method="bearer")

# Query parameter (?token=<key>) — useful when an upstream proxy strips headers.
munsit.STT(auth_method="query")
```

All three methods work on both the batch HTTP endpoint and the streaming WebSocket endpoint.

## Streaming Endpointing

When `mode="streaming"` is enabled, the plugin supports two endpointing strategies:

* `server_diff`: Keeps one long-lived WebSocket open, emits interim transcripts as cumulative text arrives, then emits a final transcript and end-of-speech after `finalize_after_silence_ms` of server silence.
* `client_vad`: Opens a WebSocket when local audio energy starts and closes it after silence. This gives stronger utterance boundaries with slightly more connection overhead.

```python theme={null}
stt = munsit.STT(
    mode="streaming",
    endpointing="server_diff",
    finalize_after_silence_ms=700,
)
```

## Synchronous Batch Recognition

When you have a recorded audio buffer (such as a voicemail or uploaded file) and do not need a live stream, call `recognize` directly. This always uses the batch HTTP endpoint regardless of the `mode` setting and returns a transcript with word-level timestamps.

```python theme={null}
from livekit import rtc
from livekit.plugins import munsit

stt = munsit.STT()
frames = [...]  # list of rtc.AudioFrame
combined = rtc.combine_audio_frames(frames)

result = await stt.recognize(combined)
print(result.alternatives[0].text)
for word in result.alternatives[0].words:
    print(f"{word.start_time:.2f}s -> {word.end_time:.2f}s  {word}")
```

## Tracking Turn Metrics

Each conversation turn carries timing data on its `ChatMessage`. Subscribe to `conversation_item_added` to read transcription delay, end-of-turn delay, and downstream LLM/TTS metrics:

```python theme={null}
from livekit.agents import ChatMessage


@session.on("conversation_item_added")
def on_item(event):
    msg = event.item
    if not isinstance(msg, ChatMessage):
        return
    m = msg.metrics or {}
    if msg.role == "user":
        td = m.get("transcription_delay")
        eou = m.get("end_of_turn_delay")
        if td is not None:
            print(f"STT delay: {td * 1000:.0f} ms")
        if eou is not None:
            print(f"EOU delay: {eou * 1000:.0f} ms")
    elif msg.role == "assistant":
        llm_ttft = m.get("llm_node_ttft")
        tts_ttfb = m.get("tts_node_ttfb")
        if llm_ttft:
            print(f"LLM TTFT: {llm_ttft * 1000:.0f} ms")
        if tts_ttfb:
            print(f"TTS TTFB: {tts_ttfb * 1000:.0f} ms")
```

All values are reported in seconds.

<Note>
  The previous `metrics_collected` event is deprecated. New integrations should use `conversation_item_added` and read metrics from `ChatMessage.metrics`.
</Note>

## Best Practices

* **Use batch mode in production.** It returns word-level timestamps and is more accurate than streaming on Arabic. Switch to streaming only when your UI needs to update before the speaker finishes.
* **Pin the model explicitly.** Pass `model="munsit"` or `model="munsit-en-ar"` instead of relying on the default, so a future plugin default change does not affect your agent.
* **Tighten the VAD for real microphones.** `silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3)` filters background noise and post-echo-cancellation residual without losing real speech under normal conditions. The default Silero thresholds are too permissive for full-duplex voice agents.
* **Use `console` mode while developing.** It runs locally with your microphone and prints transcripts to stdout, giving the fastest debugging loop. No LiveKit server required.
* **Keep the API key in the environment.** Set `MUNSIT_API_KEY` in your shell or `.env` file rather than passing it in code. The plugin reads it automatically.
* **Pair Munsit STT with Faseeh TTS for an Arabic-native loop.** Both plugins speak the same dialect register, which produces a more cohesive user experience than mixing in a non-Arabic-native TTS.

## Troubleshooting

### Missing or Invalid API Key

Verify that `MUNSIT_API_KEY` is available to the process running your agent:

```bash theme={null}
echo $MUNSIT_API_KEY
```

### No Final Transcript

Make sure your `AgentSession` includes VAD when using batch mode. Batch mode finalizes after LiveKit signals end-of-speech.

```python theme={null}
session = AgentSession(
    stt=munsit.STT(),
    vad=silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3),
    # ... llm, tts ...
)
```

### Microphone Feedback Loop

If the agent transcribes its own TTS playback and that fake transcript triggers a new conversational turn, the cause is residual audio leaking through the microphone after acoustic echo cancellation. The bilingual `munsit-en-ar` model is more sensitive to low-energy input than `munsit` is and can return text on those residuals.

**Fix.** Tighten the Silero VAD so quieter post-AEC audio does not reach STT:

```python theme={null}
vad=silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3),
```

If the loop persists, switch temporarily to `model="munsit"` to confirm the issue is model-specific, or run the agent with headphones so the speaker output cannot reach the microphone.

### Need Live Captions

Use streaming mode when your UI needs transcript updates before the speaker finishes:

```python theme={null}
stt = munsit.STT(mode="streaming", interim_results=True)
```

## Support

* **Package**: [livekit-plugins-munsit on PyPI](https://pypi.org/project/livekit-plugins-munsit/)
* **Plugin Issues**: [GitHub Issues](https://github.com/CNTXTFZCO0/livekit-plugins-munsit/issues)
* **Munsit Support**: [Schedule a Meeting](https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ1fgxJD82YNieOXrhWQ1_SfuJLthGBx7YDy3AQhdtdS7brN5Y3ZH_OKHkeIwhBTnmTZqkKjSQVI)
* **LiveKit Support**: [LiveKit Community](https://livekit.io/community)

## License

This plugin is licensed under Apache License 2.0. See [LICENSE](https://github.com/CNTXTFZCO0/livekit-plugins-munsit/blob/main/LICENSE) for details.
