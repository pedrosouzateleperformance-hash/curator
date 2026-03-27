from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from data_structures.nodes import AudioEventNode


@dataclass(frozen=True)
class AudioConfig:
    frame_ms: int = 30
    hop_ms: int = 15
    silence_threshold: float = 0.01
    speech_zcr_min: float = 0.05
    speech_zcr_max: float = 0.25


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError(f"unsupported sample width: {sampwidth}")

    audio = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    maxv = max(1.0, np.abs(audio).max())
    return audio / maxv, rate


def _frame_signal(audio: np.ndarray, sr: int, frame_ms: int, hop_ms: int) -> List[np.ndarray]:
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    out = []
    for start in range(0, max(1, len(audio) - frame_len + 1), hop):
        out.append(audio[start : start + frame_len])
    return out


def _zcr(frame: np.ndarray) -> float:
    signs = np.sign(frame)
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def classify_audio_frames(audio: np.ndarray, sr: int, config: AudioConfig | None = None) -> List[str]:
    cfg = config or AudioConfig()
    labels: List[str] = []
    for frame in _frame_signal(audio, sr, cfg.frame_ms, cfg.hop_ms):
        energy = float(np.sqrt(np.mean(frame**2)))
        zcr = _zcr(frame)
        if energy < cfg.silence_threshold:
            labels.append("silence")
        elif cfg.speech_zcr_min <= zcr <= cfg.speech_zcr_max:
            labels.append("speech")
        elif zcr < cfg.speech_zcr_min:
            labels.append("music")
        else:
            labels.append("sfx")
    return labels


def merge_audio_labels(labels: Sequence[str], sr: int, config: AudioConfig | None = None) -> List[AudioEventNode]:
    cfg = config or AudioConfig()
    hop_sec = cfg.hop_ms / 1000.0
    if not labels:
        return []
    events: List[AudioEventNode] = []
    start = 0
    current = labels[0]

    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            events.append(
                AudioEventNode(
                    id=f"audio_{len(events):04d}",
                    start_time=start * hop_sec,
                    end_time=idx * hop_sec,
                    type=current,
                    confidence=0.8,
                )
            )
            current = label
            start = idx

    events.append(
        AudioEventNode(
            id=f"audio_{len(events):04d}",
            start_time=start * hop_sec,
            end_time=len(labels) * hop_sec,
            type=current,
            confidence=0.8,
        )
    )
    return events
