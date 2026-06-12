from __future__ import annotations

from asr import load_asr_model, transcribe_audio
from config import (
    AUDIO_FILE,
    CORRECTED_DISPLAY_FILE,
    CORRECTED_PARAGRAPHS_FILE,
    CORRECTED_SEGMENTS_FILE,
    RAW_TRANSCRIPT_FILE,
)
from correction import build_correction_segments, correct_segments, load_corrector_model
from formatting import print_final_words, print_paragraphs, print_segments, print_words_by_chunks
from formatting import build_display_paragraphs
from output import save_outputs
from overlap import flatten_chunks, remove_overlaps
from utils import format_timestamp


def main() -> None:
    asr_model = load_asr_model()
    corrector_tokenizer, corrector_model, corrector_device = load_corrector_model()

    chunks = transcribe_audio(AUDIO_FILE, asr_model)

    print()
    print("=" * 80)
    print("WORDS BEFORE MERGE")
    print("=" * 80)
    print_words_by_chunks(chunks)

    chunks = remove_overlaps(chunks)
    final_words = flatten_chunks(chunks)

    print()
    print("=" * 80)
    print("FINAL WORDS AFTER MERGE")
    print("=" * 80)
    print_final_words(final_words)

    correction_segments = build_correction_segments(final_words)

    print()
    print("=" * 80)
    print("CORRECTION SEGMENTS")
    print("=" * 80)
    for segment in correction_segments:
        print(
            f"SEGMENT {segment.index:04d} | "
            f"{format_timestamp(segment.start)} - {format_timestamp(segment.end)} | "
            f"words={segment.end_word_index - segment.start_word_index} | "
            f"chars={len(segment.raw_text)}"
        )

    correction_segments = correct_segments(
        correction_segments,
        corrector_tokenizer,
        corrector_model,
        corrector_device,
    )
    display_paragraphs = build_display_paragraphs(correction_segments)

    print()
    print("=" * 80)
    print("CORRECTED SEGMENTS")
    print("=" * 80)
    print_segments(correction_segments)

    print()
    print("=" * 80)
    print("DISPLAY PARAGRAPHS")
    print("=" * 80)
    print_paragraphs(display_paragraphs)

    save_outputs(final_words, correction_segments, display_paragraphs)

    print()
    print("=" * 80)
    print(f"Saved raw transcript:               {RAW_TRANSCRIPT_FILE}")
    print(f"Saved corrected segment JSON:       {CORRECTED_SEGMENTS_FILE}")
    print(f"Saved corrected paragraph JSON:     {CORRECTED_PARAGRAPHS_FILE}")
    print(f"Saved corrected display transcript: {CORRECTED_DISPLAY_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
