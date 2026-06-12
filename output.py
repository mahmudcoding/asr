from __future__ import annotations

import json
from typing import Any

from config import (
    CORRECTED_DISPLAY_FILE,
    CORRECTED_PARAGRAPHS_FILE,
    CORRECTED_SEGMENTS_FILE,
    OUTPUT_DIR,
    RAW_TRANSCRIPT_FILE,
)
from utils import CorrectionSegment, DisplayParagraph, build_transcript, format_timestamp


def segment_to_dict(segment: CorrectionSegment) -> dict[str, Any]:
    return {
        "index": segment.index,
        "start": round(segment.start, 3),
        "end": round(segment.end, 3),
        "start_timestamp": format_timestamp(segment.start),
        "end_timestamp": format_timestamp(segment.end),
        "start_word_index": segment.start_word_index,
        "end_word_index": segment.end_word_index,
        "raw_text": segment.raw_text,
        "corrected_text": segment.corrected_text,
    }


def paragraph_to_dict(paragraph: DisplayParagraph) -> dict[str, Any]:
    return {
        "index": paragraph.index,
        "start": round(paragraph.start, 3),
        "end": round(paragraph.end, 3),
        "start_timestamp": format_timestamp(paragraph.start),
        "end_timestamp": format_timestamp(paragraph.end),
        "segment_start_index": paragraph.segment_start_index,
        "segment_end_index": paragraph.segment_end_index,
        "text": paragraph.text,
    }


def save_outputs(
    words: list[dict[str, Any]],
    segments: list[CorrectionSegment],
    paragraphs: list[DisplayParagraph],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_transcript = build_transcript(words)
    RAW_TRANSCRIPT_FILE.write_text(raw_transcript, encoding="utf-8")

    CORRECTED_SEGMENTS_FILE.write_text(
        json.dumps([segment_to_dict(segment) for segment in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    CORRECTED_PARAGRAPHS_FILE.write_text(
        json.dumps([paragraph_to_dict(paragraph) for paragraph in paragraphs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_lines = []
    for paragraph in paragraphs:
        display_lines.append(f"[{format_timestamp(paragraph.start)}] {paragraph.text}")

    CORRECTED_DISPLAY_FILE.write_text("\n\n".join(display_lines), encoding="utf-8")
