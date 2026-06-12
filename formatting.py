from __future__ import annotations

from typing import Any

from config import (
    PARAGRAPH_MAX_CHARS,
    PARAGRAPH_MAX_SECONDS,
    PARAGRAPH_MAX_WORDS,
    PARAGRAPH_MIN_SECONDS,
    PARAGRAPH_STRONG_PAUSE_SECONDS,
    PARAGRAPH_TARGET_CHARS,
    PARAGRAPH_TARGET_SECONDS,
    PARAGRAPH_TARGET_WORDS,
)
from utils import CorrectionSegment, DisplayParagraph, format_timestamp, is_sentence_like_end, normalize_spaces


def paragraph_text(segments: list[CorrectionSegment]) -> str:
    return normalize_spaces(" ".join(segment.corrected_text.strip() for segment in segments if segment.corrected_text.strip()))


def should_close_paragraph(current: list[CorrectionSegment], next_segment: CorrectionSegment | None) -> bool:
    if not current:
        return False

    start = current[0].start
    end = current[-1].end
    duration = end - start
    words = sum(len(segment.corrected_text.split()) for segment in current)
    chars = len(paragraph_text(current))

    if next_segment is None:
        return True

    pause = max(0.0, next_segment.start - current[-1].end)

    if duration >= PARAGRAPH_MIN_SECONDS and pause >= PARAGRAPH_STRONG_PAUSE_SECONDS:
        return True

    if duration >= PARAGRAPH_MAX_SECONDS:
        return True

    if words >= PARAGRAPH_MAX_WORDS:
        return True

    if chars >= PARAGRAPH_MAX_CHARS:
        return True

    if (
        duration >= PARAGRAPH_TARGET_SECONDS
        or words >= PARAGRAPH_TARGET_WORDS
        or chars >= PARAGRAPH_TARGET_CHARS
    ) and is_sentence_like_end(current[-1].corrected_text):
        return True

    return False


def build_display_paragraphs(segments: list[CorrectionSegment]) -> list[DisplayParagraph]:
    paragraphs: list[DisplayParagraph] = []
    current: list[CorrectionSegment] = []

    for i, segment in enumerate(segments):
        current.append(segment)
        next_segment = segments[i + 1] if i + 1 < len(segments) else None

        if should_close_paragraph(current, next_segment):
            paragraphs.append(
                DisplayParagraph(
                    index=len(paragraphs),
                    start=current[0].start,
                    end=current[-1].end,
                    segment_start_index=current[0].index,
                    segment_end_index=current[-1].index + 1,
                    text=paragraph_text(current),
                )
            )
            current = []

    if current:
        paragraphs.append(
            DisplayParagraph(
                index=len(paragraphs),
                start=current[0].start,
                end=current[-1].end,
                segment_start_index=current[0].index,
                segment_end_index=current[-1].index + 1,
                text=paragraph_text(current),
            )
        )

    return paragraphs


def print_words_by_chunks(chunks: list[dict[str, Any]]) -> None:
    for chunk in chunks:
        print()
        print(
            f"CHUNK {chunk['chunk_index']:04d} | "
            f"{chunk['chunk_start']:.2f} - {chunk['chunk_end']:.2f}"
        )

        for word in chunk["words"]:
            print(
                f"{word['global_start']:8.2f} -> "
                f"{word['global_end']:8.2f} | "
                f"{word['word']}"
            )


def print_final_words(words: list[dict[str, Any]]) -> None:
    for index, word in enumerate(words):
        print(
            f"{index:04d} | "
            f"{word['global_start']:8.2f} -> "
            f"{word['global_end']:8.2f} | "
            f"{word['word']}"
        )


def print_segments(segments: list[CorrectionSegment]) -> None:
    for segment in segments:
        print()
        print(f"SEGMENT {segment.index:04d} | {format_timestamp(segment.start)} - {format_timestamp(segment.end)}")
        print(f"RAW:       {segment.raw_text}")
        print(f"CORRECTED: {segment.corrected_text}")


def print_paragraphs(paragraphs: list[DisplayParagraph]) -> None:
    for paragraph in paragraphs:
        print()
        print(f"PARAGRAPH {paragraph.index:04d} | {format_timestamp(paragraph.start)} - {format_timestamp(paragraph.end)}")
        print(paragraph.text)
