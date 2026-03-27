from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from data_structures.nodes import UtteranceNode

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "is",
    "are",
    "it",
    "that",
    "this",
}
TRIGGER_LEXICON = {"love", "hate", "fear", "loss", "danger", "hope", "death", "alive", "save", "run"}
POSITIVE = {"love", "hope", "great", "good", "safe", "alive", "calm"}
NEGATIVE = {"hate", "fear", "danger", "death", "bad", "panic", "loss"}


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    speaker_id: str = "speaker_0"


def sentence_segment(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
    freq: Dict[str, int] = {}
    for token in tokens:
        if token in STOPWORDS or len(token) < 3:
            continue
        freq[token] = freq.get(token, 0) + 1
    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered[:top_k]]


def trigger_words(text: str) -> List[str]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
    return sorted({t for t in tokens if t in TRIGGER_LEXICON})


def classify_emotion(text: str) -> str:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
    pos = sum(t in POSITIVE for t in tokens)
    neg = sum(t in NEGATIVE for t in tokens)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def build_utterances(segments: Sequence[TranscriptSegment]) -> List[UtteranceNode]:
    utterances: List[UtteranceNode] = []
    for idx, seg in enumerate(segments):
        utterances.append(
            UtteranceNode(
                id=f"utt_{idx:04d}",
                text=seg.text.strip(),
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker_id=seg.speaker_id,
                keywords=extract_keywords(seg.text),
                trigger_words=trigger_words(seg.text),
                emotion_label=classify_emotion(seg.text),
            )
        )
    return utterances


def asr_from_jsonl(records: Iterable[Dict[str, object]]) -> List[TranscriptSegment]:
    segments: List[TranscriptSegment] = []
    for rec in records:
        segments.append(
            TranscriptSegment(
                text=str(rec.get("text", "")),
                start_time=float(rec.get("start_time", 0.0)),
                end_time=float(rec.get("end_time", 0.0)),
                speaker_id=str(rec.get("speaker_id", "speaker_0")),
            )
        )
    return segments
