# Proposed additions to docs.munsit.com/integrations/livekit-stt-plugin

The published page already covers Prerequisites, API Key, Installation, Quick Start, Modes, the Full Agent Example, Model Selection, Configuration Reference, Streaming Endpointing, Troubleshooting, Support, and License.

The blocks below are proposed additions/edits, written to match the existing Mintlify formatting (`theme={null}` code fences, `<Warning>` / `<Note>` callouts, plain prose tone). Each block notes where it should be inserted.

Source: real-world findings from running the v0.3.0 plugin against `munsit-en-ar` with a LiveKit `AgentSession`.

---

## Edit 1 — Update the Full Agent Example

In the existing Full Agent Example, replace this line:

```python theme={null}
vad=silero.VAD.load(),
```

with:

```python theme={null}
vad=silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3),
```

Also consider swapping `tts=openai.TTS()` for an Arabic-native option such as `cartesia.TTS()` or `faseeh.TTS()` — both produce noticeably better Arabic prosody than the OpenAI default.

**Rationale:** Real-world microphone input contains residual audio from the agent's own TTS playback (LiveKit's acoustic echo canceller reduces it but does not fully eliminate it). With default Silero thresholds, that residual can trigger an STT submission. Recommending tighter thresholds in the headline example prevents first-time users from hitting a self-reinforcing feedback loop.

---

## Addition 2 — New section between "Streaming Endpointing" and "Troubleshooting"

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

Each conversation turn carries timing data on its `ChatMessage`. Subscribe to `conversation_item_added` to read end-of-turn delay, transcription delay, and downstream LLM/TTS metrics:

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
```

`transcription_delay` and `end_of_turn_delay` are reported in seconds.

<Note>
  The previous `metrics_collected` event is deprecated. New integrations should use `conversation_item_added` and read metrics from `ChatMessage.metrics`.
</Note>

---

## Addition 3 — New entry under "Troubleshooting"

Insert this entry after "No Final Transcript".

### Microphone Feedback Loop with `munsit-en-ar`

If the agent transcribes its own TTS playback and that fake transcript triggers a new conversational turn, the cause is residual audio leaking through the microphone after acoustic echo cancellation. The bilingual `munsit-en-ar` model is more sensitive to low-energy input than `munsit` is and can return text on those residuals.

**Fix.** Tighten the Silero VAD so quieter post-AEC audio does not reach STT:

```python theme={null}
vad=silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3),
```

If the loop persists, switch temporarily to `model="munsit"` to confirm the issue is model-specific, or run the agent with headphones so the speaker output cannot reach the microphone.

---

## Addition 4 — New "Best Practices" section, just before "Troubleshooting"

## Best Practices

* **Use batch mode in production.** It returns word-level timestamps and is more accurate than streaming on Arabic. Switch to streaming only when your UI must update before the speaker finishes.
* **Pin the model explicitly.** Pass `model="munsit"` or `model="munsit-en-ar"` instead of relying on the default, so a future plugin default change does not affect your agent.
* **Tighten the VAD for real microphones.** `silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3)` filters background noise and AEC residual without losing real speech in normal conditions.
* **Use `console` mode while developing.** It runs locally with your microphone and prints transcripts to stdout, giving the fastest debugging loop.
* **Read API key from the environment.** Set `MUNSIT_API_KEY` in your shell or `.env` file rather than hard-coding it. The plugin reads it automatically.
