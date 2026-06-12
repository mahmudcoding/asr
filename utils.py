from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from config import MAX_WORD_TIME_DIFF_SECONDS, PUNCTUATION_END_RE


@dataclass
class CorrectionSegment:
    index: int
    start: float
    end: float
    start_word_index: int
    end_word_index: int
    raw_text: str
    corrected_text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class DisplayParagraph:
    index: int
    start: float
    end: float
    segment_start_index: int
    segment_end_index: int
    text: str


def normalize_apostrophes(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("ʻ", "'")
        .replace("‘", "'")
        .replace("ʼ", "'")
        .replace("´", "'")
    )


def normalize_spaces(text: str) -> str:
    text = normalize_apostrophes(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    return text.strip()


def normalize_word(word: str) -> str:
    word = normalize_apostrophes(word.lower())
    return "".join(ch for ch in word if ch.isalpha() or ch == "'")


def word_text_similarity(word1: str, word2: str) -> float:
    word1 = normalize_word(word1)
    word2 = normalize_word(word2)

    if not word1 or not word2:
        return 0.0

    if word1 == word2:
        return 1.0

    word1_no_apostrophe = word1.replace("'", "")
    word2_no_apostrophe = word2.replace("'", "")

    if word1_no_apostrophe and word1_no_apostrophe == word2_no_apostrophe:
        return 0.97

    return SequenceMatcher(None, word1, word2).ratio()


def text_similarity_for_gate(raw_text: str, corrected_text: str) -> float:
    raw_norm = re.sub(r"[^\w'А-Яа-яЁёЎўҚқҒғҲҳ]+", "", normalize_apostrophes(raw_text.lower()))
    corrected_norm = re.sub(r"[^\w'А-Яа-яЁёЎўҚқҒғҲҳ]+", "", normalize_apostrophes(corrected_text.lower()))

    if not raw_norm or not corrected_norm:
        return 0.0

    return SequenceMatcher(None, raw_norm, corrected_norm).ratio()


def word_mid_time(word: dict[str, Any]) -> float:
    return (word["global_start"] + word["global_end"]) / 2


def word_time_similarity(word1: dict[str, Any], word2: dict[str, Any]) -> float:
    diff = abs(word_mid_time(word1) - word_mid_time(word2))

    if diff > MAX_WORD_TIME_DIFF_SECONDS:
        return 0.0

    return 1.0 - (diff / MAX_WORD_TIME_DIFF_SECONDS)


def word_pause_after(words: list[dict[str, Any]], index: int) -> float:
    if index < 0 or index >= len(words) - 1:
        return 0.0
    return max(0.0, words[index + 1]["global_start"] - words[index]["global_end"])


def build_transcript(words: list[dict[str, Any]]) -> str:
    return normalize_spaces(" ".join(word["word"].strip() for word in words if word["word"].strip()))


def format_timestamp(seconds: float) -> str:
    total_seconds = int(max(0.0, seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def is_sentence_like_end(text: str) -> bool:
    return bool(PUNCTUATION_END_RE.search(text.strip()))
