from __future__ import annotations

from typing import Any

from faster_whisper import WhisperModel
from pydub import AudioSegment

from config import (
    ASR_BEAM_SIZE,
    ASR_CHUNK_OVERLAP_SECONDS,
    ASR_CHUNK_SECONDS,
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_LANGUAGE,
    ASR_MODEL_PATH,
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
