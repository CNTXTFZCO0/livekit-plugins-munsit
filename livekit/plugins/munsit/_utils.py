# Copyright 2026 LiveKit, Inc.
from __future__ import annotations

import struct
import time
from collections.abc import Callable
from enum import Enum
from typing import Generic, TypeVar

import numpy as np

from livekit import rtc


def build_wav_header(
    *, sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16
) -> bytes:
    """Build a 44-byte PCM WAV header.

    The data chunk size is set to a sentinel max value (0xFFFFFFFF - 44) because we are streaming
    and don't know the total size in advance. Munsit only validates the format fields.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")
    if bits_per_sample not in (8, 16, 24, 32):
        raise ValueError("bits_per_sample must be 8, 16, 24, or 32")

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0xFFFFFFFF - 44
    riff_size = data_size + 36

    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)  # fmt chunk size
        + struct.pack("<H", 1)  # PCM format
        + struct.pack("<H", num_channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bits_per_sample)
        + b"data"
        + struct.pack("<I", data_size)
    )


def pcm_to_audiobuffer(data: bytes) -> list[int]:
    """Encode raw PCM bytes as Munsit's `audioBuffer` JSON int array (0-255 per byte)."""
    return list(data)


T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    """Accumulate values and call ``callback`` once per ``duration`` seconds.

    Used by SpeechStream to batch audio-duration reports so RECOGNITION_USAGE
    events don't fire on every push.
    """

    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore[operator]
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


_DEFAULT_RMS_THRESHOLD = 0.004**2


class AudioEnergyFilter:
    """Simple RMS-based VAD copied from the Gladia plugin pattern."""

    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(
        self, *, min_silence: float = 1.5, rms_threshold: float = _DEFAULT_RMS_THRESHOLD
    ) -> None:
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> AudioEnergyFilter.State:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = float(np.mean(np.square(float_arr)))

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING

        return self._state
