from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    CORRECTION_FORCE_SECONDS,
    CORRECTION_GOOD_PAUSE_SECONDS,
    CORRECTION_MAX_CHARS,
    CORRECTION_MAX_SECONDS,
    CORRECTION_MAX_WORDS,
    CORRECTION_MIN_SECONDS,
    CORRECTION_MIN_WORDS,
    CORRECTION_STRONG_PAUSE_SECONDS,
    CORRECTION_TARGET_CHARS,
    CORRECTION_TARGET_SECONDS,
    CORRECTION_TARGET_WORDS,
    CORRECTOR_DEVICE,
    CORRECTOR_MAX_INPUT_TOKENS,
    CORRECTOR_MAX_NEW_TOKENS,
    CORRECTOR_MODEL_PATH,
    CORRECTOR_NUM_BEAMS,
    WEAK_END_WORDS,
)
from utils import (
    CorrectionSegment,
    build_transcript,
    format_timestamp,
    is_sentence_like_end,
    normalize_spaces,
    normalize_word,
    text_similarity_for_gate,
    word_pause_after,
)


def resolve_corrector_device() -> torch.device:
    if CORRECTOR_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(CORRECTOR_DEVICE)


def load_corrector_model() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]:
    print("Loading transcript corrector model...")
    device = resolve_corrector_device()
    tokenizer = AutoTokenizer.from_pretrained(CORRECTOR_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTOR_MODEL_PATH)
    model.to(device)
    model.eval()
    print(f"Corrector device: {device}")
    return tokenizer, model, device


def boundary_score(words: list[dict[str, Any]], start_index: int, cut_index: int) -> float:
    """
    Scores a possible boundary after cut_index, where the segment is
    words[start_index:cut_index + 1]. Higher is better.
    """
    segment_words = words[start_index:cut_index + 1]
    first_word = segment_words[0]
    last_word = segment_words[-1]

    duration = max(0.0, last_word["global_end"] - first_word["global_start"])
    word_count = len(segment_words)
    char_count = len(build_transcript(segment_words))
    pause = word_pause_after(words, cut_index)

    last_text = normalize_word(str(last_word["word"]))
    next_text = normalize_word(str(words[cut_index + 1]["word"])) if cut_index + 1 < len(words) else ""

    score = 0.0

    if pause >= CORRECTION_STRONG_PAUSE_SECONDS:
        score += 8.0
    elif pause >= CORRECTION_GOOD_PAUSE_SECONDS:
        score += 5.0
    elif pause >= 0.35:
        score += 2.0

    score -= abs(duration - CORRECTION_TARGET_SECONDS) * 0.35
    score -= abs(word_count - CORRECTION_TARGET_WORDS) * 0.08
    score -= abs(char_count - CORRECTION_TARGET_CHARS) * 0.01

    if is_sentence_like_end(str(last_word["word"])):
        score += 2.0

    if last_text in WEAK_END_WORDS:
        score -= 3.0

    if next_text in WEAK_END_WORDS:
        score -= 1.0

    if duration < CORRECTION_MIN_SECONDS:
        score -= 5.0
    if word_count < CORRECTION_MIN_WORDS:
        score -= 4.0

    if duration > CORRECTION_MAX_SECONDS:
        score -= (duration - CORRECTION_MAX_SECONDS) * 2.0
    if word_count > CORRECTION_MAX_WORDS:
        score -= (word_count - CORRECTION_MAX_WORDS) * 0.8
    if char_count > CORRECTION_MAX_CHARS:
        score -= (char_count - CORRECTION_MAX_CHARS) * 0.05

    return score


def choose_correction_cut(words: list[dict[str, Any]], start_index: int) -> int:
    if start_index >= len(words) - 1:
        return len(words) - 1

    best_cut = start_index
    best_score = float("-inf")

    for cut_index in range(start_index, len(words)):
        segment_words = words[start_index:cut_index + 1]
        duration = segment_words[-1]["global_end"] - segment_words[0]["global_start"]
        word_count = len(segment_words)
        char_count = len(build_transcript(segment_words))
        pause = word_pause_after(words, cut_index)

        can_cut = duration >= CORRECTION_MIN_SECONDS and word_count >= CORRECTION_MIN_WORDS

        if can_cut and pause >= CORRECTION_STRONG_PAUSE_SECONDS:
            return cut_index

        if can_cut:
            score = boundary_score(words, start_index, cut_index)
            if score > best_score:
                best_score = score
                best_cut = cut_index

        force_cut = (
            duration >= CORRECTION_FORCE_SECONDS
            or word_count >= CORRECTION_MAX_WORDS
            or char_count >= CORRECTION_MAX_CHARS
        )

        if force_cut:
            if best_cut > start_index:
                return best_cut
            return cut_index

    return len(words) - 1


def build_correction_segments(words: list[dict[str, Any]]) -> list[CorrectionSegment]:
    segments: list[CorrectionSegment] = []
    start_index = 0

    while start_index < len(words):
        cut_index = choose_correction_cut(words, start_index)

        segment_words = words[start_index:cut_index + 1]
        if not segment_words:
            break

        segment = CorrectionSegment(
            index=len(segments),
            start=float(segment_words[0]["global_start"]),
            end=float(segment_words[-1]["global_end"]),
            start_word_index=start_index,
            end_word_index=cut_index + 1,
            raw_text=build_transcript(segment_words),
        )
        segments.append(segment)

        start_index = cut_index + 1

    # Attach tiny final segment to previous segment.
    if len(segments) >= 2:
        last = segments[-1]
        previous = segments[-2]
        last_word_count = last.end_word_index - last.start_word_index

        if last.duration < 2.0 or last_word_count < 5:
            merged_words = words[previous.start_word_index:last.end_word_index]
            segments[-2] = CorrectionSegment(
                index=previous.index,
                start=previous.start,
                end=last.end,
                start_word_index=previous.start_word_index,
                end_word_index=last.end_word_index,
                raw_text=build_transcript(merged_words),
            )
            segments.pop()

    for i, segment in enumerate(segments):
        segment.index = i

    return segments


def correct_text(
    raw_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
) -> str:
    raw_text = normalize_spaces(raw_text)

    if not raw_text:
        return ""

    prompt = f"correct: {raw_text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CORRECTOR_MAX_INPUT_TOKENS,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CORRECTOR_MAX_NEW_TOKENS,
            num_beams=CORRECTOR_NUM_BEAMS,
            do_sample=False,
        )

    corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    corrected = normalize_spaces(corrected)

    # Conservative safety gate: never allow broken model output to destroy text.
    if not corrected:
        return raw_text

    if len(corrected) > max(80, len(raw_text) * 3):
        print("Warning: correction output was suspiciously long. Using raw text for this segment.")
        return raw_text

    similarity = text_similarity_for_gate(raw_text, corrected)
    if similarity < 0.25:
        print("Warning: correction output is very different from raw segment. Review this segment manually.")

    return corrected


def correct_segments(
    segments: list[CorrectionSegment],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
) -> list[CorrectionSegment]:
    for segment in segments:
        print(
            f"Correcting segment {segment.index:04d}: "
            f"{format_timestamp(segment.start)} - {format_timestamp(segment.end)} | "
            f"{segment.end_word_index - segment.start_word_index} words"
        )
        segment.corrected_text = correct_text(segment.raw_text, tokenizer, model, device)

    return segments
