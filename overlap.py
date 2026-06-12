from __future__ import annotations

from typing import Any

from config import (
    ASR_CHUNK_OVERLAP_SAFE_PADDING_SECONDS,
    MAX_WORD_TIME_DIFF_SECONDS,
    MIN_MATCHED_WORDS,
    TEXT_SIMILARITY_THRESHOLD,
)
from utils import word_mid_time, word_text_similarity, word_time_similarity


def words_are_matchable(word1: dict[str, Any], word2: dict[str, Any]) -> bool:
    text_sim = word_text_similarity(word1["word"], word2["word"])

    if text_sim < TEXT_SIMILARITY_THRESHOLD:
        return False

    time_diff = abs(word_mid_time(word1) - word_mid_time(word2))

    if time_diff > MAX_WORD_TIME_DIFF_SECONDS:
        return False

    return True


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
