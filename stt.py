from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# =============================================================================
# INPUT / OUTPUT
# =============================================================================

AUDIO_FILE = "audio_short.wav"

OUTPUT_DIR = Path("output")
CHUNKS_DIR = Path("chunks")

RAW_TRANSCRIPT_FILE = OUTPUT_DIR / "transcript_raw.txt"
CORRECTED_SEGMENTS_FILE = OUTPUT_DIR / "transcript_corrected_segments.json"
CORRECTED_PARAGRAPHS_FILE = OUTPUT_DIR / "transcript_corrected_paragraphs.json"
CORRECTED_DISPLAY_FILE = OUTPUT_DIR / "transcript_corrected_display.txt"


# =============================================================================
# ASR MODEL
# =============================================================================

ASR_MODEL_PATH = "kotib_ct2"
ASR_DEVICE = "cpu"
ASR_COMPUTE_TYPE = "int8"
ASR_LANGUAGE = "uz"
ASR_BEAM_SIZE = 1

# Audio is sent to Whisper in overlapping chunks. This is not the same thing as
# correction/display segmentation.
ASR_CHUNK_SECONDS = 30
ASR_CHUNK_OVERLAP_SECONDS = 5
ASR_CHUNK_OVERLAP_SAFE_PADDING_SECONDS = 0.75


# =============================================================================
# TRANSCRIPT CORRECTOR MODEL
# =============================================================================

CORRECTOR_MODEL_PATH = "islomov/rubai-corrector-transcript-uz"

# Use "auto" for CUDA if available, otherwise CPU.
# On your Mac / CPU setup, it will automatically use CPU.
CORRECTOR_DEVICE = "auto"

# ByT5 is byte-level, so do not send very large text. Correction segments below
# are intentionally short.
CORRECTOR_MAX_INPUT_TOKENS = 1024
CORRECTOR_MAX_NEW_TOKENS = 512
CORRECTOR_NUM_BEAMS = 1


# =============================================================================
# OVERLAP MERGE SETTINGS
# =============================================================================

MIN_MATCHED_WORDS = 3
TEXT_SIMILARITY_THRESHOLD = 0.82
MAX_WORD_TIME_DIFF_SECONDS = 1.75


# =============================================================================
# CORRECTION SEGMENT SETTINGS
# =============================================================================
# These chunks are for the transcript corrector, not Whisper.
# They are based on speech timing, pauses, and safe max limits.

CORRECTION_MIN_SECONDS = 3.0
CORRECTION_TARGET_SECONDS = 8.0
CORRECTION_MAX_SECONDS = 12.0
CORRECTION_FORCE_SECONDS = 14.0

CORRECTION_MIN_WORDS = 8
CORRECTION_TARGET_WORDS = 24
CORRECTION_MAX_WORDS = 40

CORRECTION_TARGET_CHARS = 180
CORRECTION_MAX_CHARS = 280

CORRECTION_GOOD_PAUSE_SECONDS = 0.70
CORRECTION_STRONG_PAUSE_SECONDS = 1.20


# =============================================================================
# DISPLAY PARAGRAPH SETTINGS
# =============================================================================
# Paragraphs are for user display. They are bigger than correction segments.

PARAGRAPH_MIN_SECONDS = 12.0
PARAGRAPH_TARGET_SECONDS = 28.0
PARAGRAPH_MAX_SECONDS = 45.0

PARAGRAPH_TARGET_WORDS = 80
PARAGRAPH_MAX_WORDS = 130
PARAGRAPH_TARGET_CHARS = 550
PARAGRAPH_MAX_CHARS = 900
PARAGRAPH_STRONG_PAUSE_SECONDS = 2.0


# =============================================================================
# SMALL LANGUAGE HEURISTICS FOR SAFER BOUNDARIES
# =============================================================================

WEAK_END_WORDS = {
    "va", "yoki", "ham", "lekin", "ammo", "bilan", "uchun", "agar", "chunki",
    "ki", "bu", "shu", "o'sha", "mana", "ya'ni", "ya'ni", "masalan",
    "и", "а", "но", "или", "что", "если", "потому", "потому что", "для", "с",
    "and", "or", "but", "if", "because", "for", "with", "to", "of",
}

PUNCTUATION_END_RE = re.compile(r"[.!?…]+[\"')\]]*$")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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


# =============================================================================
# MODEL LOADING
# =============================================================================

print("Loading ASR model...")
asr_model = WhisperModel(
    ASR_MODEL_PATH,
    device=ASR_DEVICE,
    compute_type=ASR_COMPUTE_TYPE,
)


def resolve_corrector_device() -> torch.device:
    if CORRECTOR_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(CORRECTOR_DEVICE)


print("Loading transcript corrector model...")
corrector_device = resolve_corrector_device()
corrector_tokenizer = AutoTokenizer.from_pretrained(CORRECTOR_MODEL_PATH)
corrector_model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTOR_MODEL_PATH)
corrector_model.to(corrector_device)
corrector_model.eval()
print(f"Corrector device: {corrector_device}")


# =============================================================================
# BASIC TEXT NORMALIZATION
# =============================================================================

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


# =============================================================================
# WORD TIME HELPERS
# =============================================================================

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


# =============================================================================
# ASR OVERLAP MERGE
# =============================================================================

def words_are_matchable(word1: dict[str, Any], word2: dict[str, Any]) -> bool:
    text_sim = word_text_similarity(word1["word"], word2["word"])

    if text_sim < TEXT_SIMILARITY_THRESHOLD:
        return False

    time_diff = abs(word_mid_time(word1) - word_mid_time(word2))

    if time_diff > MAX_WORD_TIME_DIFF_SECONDS:
        return False

    return True


def merge_apostrophe_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    i = 0

    while i < len(words):
        current = words[i]

        if i + 1 < len(words):
            next_word = words[i + 1]
            next_text = normalize_apostrophes(next_word["word"]).strip()

            if next_text.startswith("'"):
                merged.append(
                    {
                        "local_start": current["local_start"],
                        "local_end": next_word["local_end"],
                        "global_start": current["global_start"],
                        "global_end": next_word["global_end"],
                        "word": current["word"] + next_word["word"],
                    }
                )
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged


def transcribe_audio(audio_file: str) -> list[dict[str, Any]]:
    audio = AudioSegment.from_file(audio_file)

    CHUNKS_DIR.mkdir(exist_ok=True)

    chunk_ms = ASR_CHUNK_SECONDS * 1000
    overlap_ms = ASR_CHUNK_OVERLAP_SECONDS * 1000
    step_ms = chunk_ms - overlap_ms

    if step_ms <= 0:
        raise ValueError("ASR_CHUNK_SECONDS must be larger than ASR_CHUNK_OVERLAP_SECONDS")

    duration_ms = len(audio)
    chunks = []

    for chunk_index, start_ms in enumerate(range(0, duration_ms, step_ms)):
        end_ms = min(start_ms + chunk_ms, duration_ms)

        if start_ms >= end_ms:
            break

        chunk_audio = audio[start_ms:end_ms]
        chunk_file = CHUNKS_DIR / f"chunk_{chunk_index:04d}_{start_ms // 1000}-{end_ms // 1000}.wav"
        chunk_audio.export(chunk_file, format="wav")

        chunk_start_seconds = start_ms / 1000
        chunk_end_seconds = end_ms / 1000

        print(f"Transcribing chunk {chunk_index:04d}: {chunk_start_seconds:.2f} - {chunk_end_seconds:.2f}")

        segments, info = asr_model.transcribe(
            str(chunk_file),
            language=ASR_LANGUAGE,
            task="transcribe",
            beam_size=ASR_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=False,
        )

        words = []

        for segment in segments:
            if segment.words is None:
                continue

            for word in segment.words:
                if word.start is None or word.end is None:
                    continue

                local_start = float(word.start)
                local_end = float(word.end)

                words.append(
                    {
                        "local_start": local_start,
                        "local_end": local_end,
                        "global_start": chunk_start_seconds + local_start,
                        "global_end": chunk_start_seconds + local_end,
                        "word": word.word,
                    }
                )

        words = merge_apostrophe_words(words)

        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_start": chunk_start_seconds,
                "chunk_end": chunk_end_seconds,
                "words": words,
            }
        )

    return chunks


def get_overlap_words(
    first_chunk: dict[str, Any],
    second_chunk: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    overlap_start = second_chunk["chunk_start"]
    overlap_end = first_chunk["chunk_end"]

    padded_overlap_start = overlap_start - ASR_CHUNK_OVERLAP_SAFE_PADDING_SECONDS
    padded_overlap_end = overlap_end + ASR_CHUNK_OVERLAP_SAFE_PADDING_SECONDS

    first_words = first_chunk["words"]
    second_words = second_chunk["words"]

    first_overlap_start_index = len(first_words)

    for i, word in enumerate(first_words):
        if word["global_end"] >= padded_overlap_start:
            first_overlap_start_index = i
            break

    second_overlap_end_index = 0

    for i, word in enumerate(second_words):
        if word["global_start"] <= padded_overlap_end:
            second_overlap_end_index = i + 1
        else:
            break

    first_overlap = first_words[first_overlap_start_index:]
    second_overlap = second_words[:second_overlap_end_index]

    return first_overlap, second_overlap, first_overlap_start_index, second_overlap_end_index


def local_align_overlap(
    first_words: list[dict[str, Any]],
    second_words: list[dict[str, Any]],
) -> list[tuple[int, int, float, float]]:
    n = len(first_words)
    m = len(second_words)

    if n == 0 or m == 0:
        return []

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    direction = [[None] * (m + 1) for _ in range(n + 1)]

    best_score = 0.0
    best_pos = (0, 0)

    gap_penalty = -0.75
    mismatch_penalty = -1.25

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            word1 = first_words[i - 1]
            word2 = second_words[j - 1]

            text_sim = word_text_similarity(word1["word"], word2["word"])
            time_sim = word_time_similarity(word1, word2)

            if words_are_matchable(word1, word2):
                match_score = 2.5 * text_sim + 1.5 * time_sim
            else:
                match_score = mismatch_penalty

            diagonal = score[i - 1][j - 1] + match_score
            up = score[i - 1][j] + gap_penalty
            left = score[i][j - 1] + gap_penalty

            best = max(0.0, diagonal, up, left)
            score[i][j] = best

            if best == 0.0:
                direction[i][j] = None
            elif best == diagonal:
                direction[i][j] = "diag"
            elif best == up:
                direction[i][j] = "up"
            else:
                direction[i][j] = "left"

            if best > best_score:
                best_score = best
                best_pos = (i, j)

    alignment = []
    i, j = best_pos

    while i > 0 and j > 0 and score[i][j] > 0:
        move = direction[i][j]

        if move == "diag":
            word1 = first_words[i - 1]
            word2 = second_words[j - 1]

            text_sim = word_text_similarity(word1["word"], word2["word"])
            time_diff = abs(word_mid_time(word1) - word_mid_time(word2))

            alignment.append((i - 1, j - 1, text_sim, time_diff))
            i -= 1
            j -= 1
        elif move == "up":
            i -= 1
        elif move == "left":
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment


def extract_best_matching_run(
    alignment: list[tuple[int, int, float, float]],
) -> list[tuple[int, int, float, float]]:
    best_run = []
    current_run = []

    previous_i = None
    previous_j = None

    for item in alignment:
        i, j, text_sim, time_diff = item

        is_good_match = text_sim >= TEXT_SIMILARITY_THRESHOLD and time_diff <= MAX_WORD_TIME_DIFF_SECONDS
        is_continuation = (
            previous_i is None
            or (
                i > previous_i
                and j > previous_j
                and i - previous_i <= 2
                and j - previous_j <= 2
            )
        )

        if is_good_match and is_continuation:
            current_run.append(item)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = [item] if is_good_match else []

        previous_i = i
        previous_j = j

    if len(current_run) > len(best_run):
        best_run = current_run

    return best_run


def is_reliable_match(match_run: list[tuple[int, int, float, float]]) -> bool:
    if len(match_run) < MIN_MATCHED_WORDS:
        return False

    avg_text_similarity = sum(item[2] for item in match_run) / len(match_run)
    avg_time_diff = sum(item[3] for item in match_run) / len(match_run)

    if avg_text_similarity < TEXT_SIMILARITY_THRESHOLD:
        return False

    if avg_time_diff > MAX_WORD_TIME_DIFF_SECONDS:
        return False

    return True


def remove_overlaps(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for chunk_index in range(len(chunks) - 1):
        first_chunk = chunks[chunk_index]
        second_chunk = chunks[chunk_index + 1]

        first_words = first_chunk["words"]
        second_words = second_chunk["words"]

        if not first_words or not second_words:
            continue

        first_overlap, second_overlap, first_overlap_start_index, _ = get_overlap_words(first_chunk, second_chunk)

        if not first_overlap or not second_overlap:
            continue

        alignment = local_align_overlap(first_overlap, second_overlap)
        best_run = extract_best_matching_run(alignment)

        if not is_reliable_match(best_run):
            print(
                f"No reliable overlap match between chunk "
                f"{first_chunk['chunk_index']} and {second_chunk['chunk_index']}. Keeping both."
            )
            continue

        first_match_start_in_overlap = best_run[0][0]
        second_match_start_in_overlap = best_run[0][1]

        first_cut_index = first_overlap_start_index + first_match_start_in_overlap
        second_cut_index = second_match_start_in_overlap

        first_chunk["words"] = first_words[:first_cut_index]
        second_chunk["words"] = second_words[second_cut_index:]

        print(
            f"Merged chunk {first_chunk['chunk_index']} -> {second_chunk['chunk_index']} | "
            f"matched_words={len(best_run)} | "
            f"first_cut={first_cut_index} | "
            f"second_cut={second_cut_index}"
        )

    return chunks


def flatten_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words = []

    for chunk in chunks:
        words.extend(chunk["words"])

    words.sort(key=lambda w: (w["global_start"], w["global_end"]))
    return words


# =============================================================================
# TRANSCRIPT BUILDING
# =============================================================================

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


# =============================================================================
# SMART CORRECTION SEGMENTATION
# =============================================================================

def is_sentence_like_end(text: str) -> bool:
    return bool(PUNCTUATION_END_RE.search(text.strip()))


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

    # Pause is the strongest natural boundary signal.
    if pause >= CORRECTION_STRONG_PAUSE_SECONDS:
        score += 8.0
    elif pause >= CORRECTION_GOOD_PAUSE_SECONDS:
        score += 5.0
    elif pause >= 0.35:
        score += 2.0

    # Prefer target length, but allow natural variation.
    score -= abs(duration - CORRECTION_TARGET_SECONDS) * 0.35
    score -= abs(word_count - CORRECTION_TARGET_WORDS) * 0.08
    score -= abs(char_count - CORRECTION_TARGET_CHARS) * 0.01

    # Whisper punctuation is weak, but useful when available.
    if is_sentence_like_end(str(last_word["word"])):
        score += 2.0

    # Avoid ending immediately after connectors/prepositions.
    if last_text in WEAK_END_WORDS:
        score -= 3.0

    # Starting a new segment with a connector is not ideal, but less bad than
    # cutting after a connector.
    if next_text in WEAK_END_WORDS:
        score -= 1.0

    # Very short segments are bad unless there is a strong pause.
    if duration < CORRECTION_MIN_SECONDS:
        score -= 5.0
    if word_count < CORRECTION_MIN_WORDS:
        score -= 4.0

    # Strongly discourage too-long segments.
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

        can_cut = (
            duration >= CORRECTION_MIN_SECONDS
            and word_count >= CORRECTION_MIN_WORDS
        )

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

    # If the last segment is tiny, attach it to the previous one. This improves
    # correction quality and avoids ugly display fragments.
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


# =============================================================================
# TRANSCRIPT CORRECTION
# =============================================================================

def correct_text(raw_text: str) -> str:
    raw_text = normalize_spaces(raw_text)

    if not raw_text:
        return ""

    prompt = f"correct: {raw_text}"

    inputs = corrector_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CORRECTOR_MAX_INPUT_TOKENS,
    )
    inputs = {key: value.to(corrector_device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = corrector_model.generate(
            **inputs,
            max_new_tokens=CORRECTOR_MAX_NEW_TOKENS,
            num_beams=CORRECTOR_NUM_BEAMS,
            do_sample=False,
        )

    corrected = corrector_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    corrected = normalize_spaces(corrected)

    # Safety gate: never allow a broken correction model output to destroy the
    # transcript. This is intentionally conservative.
    if not corrected:
        return raw_text

    if len(corrected) > max(80, len(raw_text) * 3):
        print("Warning: correction output was suspiciously long. Using raw text for this segment.")
        return raw_text

    # If the output is extremely different and not caused by expected Russian
    # recovery / punctuation, warn but keep it. The Rubai model can convert
    # latin Russian to Cyrillic, so strict similarity would be wrong.
    similarity = text_similarity_for_gate(raw_text, corrected)
    if similarity < 0.25:
        print("Warning: correction output is very different from raw segment. Review this segment manually.")

    return corrected


def correct_segments(segments: list[CorrectionSegment]) -> list[CorrectionSegment]:
    for segment in segments:
        print(
            f"Correcting segment {segment.index:04d}: "
            f"{format_timestamp(segment.start)} - {format_timestamp(segment.end)} | "
            f"{segment.end_word_index - segment.start_word_index} words"
        )
        segment.corrected_text = correct_text(segment.raw_text)

    return segments


# =============================================================================
# DISPLAY PARAGRAPH BUILDING
# =============================================================================

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

    # If close to target and current corrected text ends with punctuation, close.
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


# =============================================================================
# SAVING OUTPUTS
# =============================================================================

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


# =============================================================================
# DEBUG PRINTING
# =============================================================================

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


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    chunks = transcribe_audio(AUDIO_FILE)

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

    correction_segments = correct_segments(correction_segments)
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
    print(f"Saved raw transcript:              {RAW_TRANSCRIPT_FILE}")
    print(f"Saved corrected segment JSON:      {CORRECTED_SEGMENTS_FILE}")
    print(f"Saved corrected paragraph JSON:    {CORRECTED_PARAGRAPHS_FILE}")
    print(f"Saved corrected display transcript:{CORRECTED_DISPLAY_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
