from pathlib import Path
from difflib import SequenceMatcher
from pydub import AudioSegment
from faster_whisper import WhisperModel


AUDIO_FILE = "audio_short.wav"

MODEL_PATH = "kotib_ct2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

CHUNK_SECONDS = 30
CHUNK_OVERLAP_SECONDS = 5
CHUNK_OVERLAP_SAFE_PADDING_SECONDS = 0.75

LANGUAGE = "uz"
BEAM_SIZE = 1

CHUNKS_DIR = Path("chunks")
OUTPUT_TRANSCRIPT_FILE = "transcript.txt"

MIN_MATCHED_WORDS = 3
TEXT_SIMILARITY_THRESHOLD = 0.82
MAX_WORD_TIME_DIFF_SECONDS = 1.75


model = WhisperModel(
    MODEL_PATH,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)


def normalize_apostrophes(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("ʻ", "'")
        .replace("‘", "'")
        .replace("ʼ", "'")
    )


def normalize_word(word: str) -> str:
    word = normalize_apostrophes(word.lower())

    return "".join(
        ch
        for ch in word
        if ch.isalpha() or ch == "'"
    )


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


def word_mid_time(word: dict) -> float:
    return (word["global_start"] + word["global_end"]) / 2


def word_time_similarity(word1: dict, word2: dict) -> float:
    diff = abs(word_mid_time(word1) - word_mid_time(word2))

    if diff > MAX_WORD_TIME_DIFF_SECONDS:
        return 0.0

    return 1.0 - (diff / MAX_WORD_TIME_DIFF_SECONDS)


def words_are_matchable(word1: dict, word2: dict) -> bool:
    text_sim = word_text_similarity(word1["word"], word2["word"])

    if text_sim < TEXT_SIMILARITY_THRESHOLD:
        return False

    time_diff = abs(word_mid_time(word1) - word_mid_time(word2))

    if time_diff > MAX_WORD_TIME_DIFF_SECONDS:
        return False

    return True


def merge_apostrophe_words(words: list[dict]) -> list[dict]:
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


def transcribe_audio(audio_file: str) -> list[dict]:
    audio = AudioSegment.from_file(audio_file)

    CHUNKS_DIR.mkdir(exist_ok=True)

    chunk_ms = CHUNK_SECONDS * 1000
    overlap_ms = CHUNK_OVERLAP_SECONDS * 1000
    step_ms = chunk_ms - overlap_ms

    if step_ms <= 0:
        raise ValueError("CHUNK_SECONDS must be larger than CHUNK_OVERLAP_SECONDS")

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

        segments, info = model.transcribe(
            str(chunk_file),
            language=LANGUAGE,
            task="transcribe",
            beam_size=BEAM_SIZE,
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
    first_chunk: dict,
    second_chunk: dict,
) -> tuple[list[dict], list[dict], int, int]:
    overlap_start = second_chunk["chunk_start"]
    overlap_end = first_chunk["chunk_end"]

    padded_overlap_start = overlap_start - CHUNK_OVERLAP_SAFE_PADDING_SECONDS
    padded_overlap_end = overlap_end + CHUNK_OVERLAP_SAFE_PADDING_SECONDS

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

    return (
        first_overlap,
        second_overlap,
        first_overlap_start_index,
        second_overlap_end_index,
    )


def local_align_overlap(first_words: list[dict], second_words: list[dict]) -> list[tuple[int, int, float, float]]:
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

            alignment.append(
                (
                    i - 1,
                    j - 1,
                    text_sim,
                    time_diff,
                )
            )

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
    alignment: list[tuple[int, int, float, float]]
) -> list[tuple[int, int, float, float]]:
    best_run = []
    current_run = []

    previous_i = None
    previous_j = None

    for item in alignment:
        i, j, text_sim, time_diff = item

        is_good_match = (
            text_sim >= TEXT_SIMILARITY_THRESHOLD
            and time_diff <= MAX_WORD_TIME_DIFF_SECONDS
        )

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


def remove_overlaps(chunks: list[dict]) -> list[dict]:
    for chunk_index in range(len(chunks) - 1):
        first_chunk = chunks[chunk_index]
        second_chunk = chunks[chunk_index + 1]

        first_words = first_chunk["words"]
        second_words = second_chunk["words"]

        if not first_words or not second_words:
            continue

        (
            first_overlap,
            second_overlap,
            first_overlap_start_index,
            second_overlap_end_index,
        ) = get_overlap_words(first_chunk, second_chunk)

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


def flatten_chunks(chunks: list[dict]) -> list[dict]:
    words = []

    for chunk in chunks:
        words.extend(chunk["words"])

    words.sort(key=lambda w: (w["global_start"], w["global_end"]))

    return words


def build_transcript(words: list[dict]) -> str:
    transcript_parts = []

    for word in words:
        text = word["word"].strip()

        if not text:
            continue

        transcript_parts.append(text)

    return " ".join(transcript_parts)


def print_words_by_chunks(chunks: list[dict]) -> None:
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


def print_final_words(words: list[dict]) -> None:
    for index, word in enumerate(words):
        print(
            f"{index:04d} | "
            f"{word['global_start']:8.2f} -> "
            f"{word['global_end']:8.2f} | "
            f"{word['word']}"
        )


def save_transcript(words: list[dict], output_file: str) -> None:
    transcript = build_transcript(words)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(transcript)


def main():
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

    save_transcript(final_words, OUTPUT_TRANSCRIPT_FILE)

    print()
    print("=" * 80)
    print(f"Saved final transcript to: {OUTPUT_TRANSCRIPT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()