from __future__ import annotations

from typing import Any

from faster_whisper import WhisperModel
from pydub import AudioSegment

from config import (
    ASR_BEAM_SIZE,
    ASR_CHUNK_OVERLAP_SECONDS,
    ASR_CHUNK_SECONDS,
    ASR_COMPUTE_TYPE,
    ASR_COMPRESSION_RATIO_THRESHOLD,
    ASR_CONDITION_ON_PREVIOUS_TEXT,
    ASR_DEVICE,
    ASR_LANGUAGE,
    ASR_LOG_PROB_THRESHOLD,
    ASR_MAX_SAME_TIMESTAMP_WORDS,
    ASR_MIN_WORD_DURATION_SECONDS,
    ASR_MODEL_PATH,
    ASR_NO_SPEECH_THRESHOLD,
    ASR_SAME_TIMESTAMP_EPSILON_SECONDS,
    ASR_VAD_FILTER,
    CHUNKS_DIR,
)
from utils import normalize_apostrophes


def load_asr_model() -> WhisperModel:
    print("Loading ASR model...")
    return WhisperModel(
        ASR_MODEL_PATH,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
    )


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


def is_punctuation_only(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    return all(not ch.isalnum() for ch in cleaned)


def filter_suspicious_words(words: list[dict[str, Any]], chunk_end_seconds: float) -> list[dict[str, Any]]:
    """
    Faster-Whisper can sometimes hallucinate words at the end of short/silent chunks,
    especially with zero-duration timestamps such as 69.58 -> 69.58 repeated several
    times. For production transcripts, it is safer to drop impossible timestamp words
    than to keep invented text.
    """
    filtered: list[dict[str, Any]] = []

    for word in words:
        text = str(word.get("word", ""))
        start = float(word["global_start"])
        end = float(word["global_end"])
        duration = end - start

        if is_punctuation_only(text):
            continue

        if end <= start:
            print(f"Dropped suspicious zero-duration ASR word: {start:.2f}->{end:.2f} | {text}")
            continue

        if duration < ASR_MIN_WORD_DURATION_SECONDS:
            print(f"Dropped suspicious ultra-short ASR word: {start:.2f}->{end:.2f} | {text}")
            continue

        # If a word starts beyond the exported chunk end because of timestamp bugs,
        # do not keep it.
        if start > chunk_end_seconds + 0.05:
            print(f"Dropped out-of-chunk ASR word: {start:.2f}->{end:.2f} | {text}")
            continue

        filtered.append(word)

    if len(filtered) <= ASR_MAX_SAME_TIMESTAMP_WORDS:
        return filtered

    cleaned: list[dict[str, Any]] = []
    i = 0

    while i < len(filtered):
        cluster = [filtered[i]]
        j = i + 1

        while j < len(filtered):
            same_start = abs(filtered[j]["global_start"] - filtered[i]["global_start"]) <= ASR_SAME_TIMESTAMP_EPSILON_SECONDS
            same_end = abs(filtered[j]["global_end"] - filtered[i]["global_end"]) <= ASR_SAME_TIMESTAMP_EPSILON_SECONDS
            if not (same_start and same_end):
                break
            cluster.append(filtered[j])
            j += 1

        if len(cluster) > ASR_MAX_SAME_TIMESTAMP_WORDS:
            joined = " ".join(str(item["word"]).strip() for item in cluster)
            print(
                f"Dropped suspicious same-timestamp ASR cluster: "
                f"{cluster[0]['global_start']:.2f}->{cluster[0]['global_end']:.2f} | {joined}"
            )
        else:
            cleaned.extend(cluster)

        i = j

    return cleaned


def transcribe_audio(audio_file: str, model: WhisperModel) -> list[dict[str, Any]]:
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

        segments, _ = model.transcribe(
            str(chunk_file),
            language=ASR_LANGUAGE,
            task="transcribe",
            beam_size=ASR_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=ASR_VAD_FILTER,
            condition_on_previous_text=ASR_CONDITION_ON_PREVIOUS_TEXT,
            no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
            log_prob_threshold=ASR_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=ASR_COMPRESSION_RATIO_THRESHOLD,
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

        words = filter_suspicious_words(words, chunk_end_seconds)
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
