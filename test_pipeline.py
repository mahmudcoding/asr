from __future__ import annotations

import unittest

from overlap import remove_overlaps
from utils import normalize_spaces, polish_transcript_text


def word(start: float, end: float, text: str) -> dict[str, float | str]:
    return {
        "local_start": start,
        "local_end": end,
        "global_start": start,
        "global_end": end,
        "word": text,
    }


class OverlapMergeTests(unittest.TestCase):
    def test_prefers_complete_word_away_from_second_chunk_start(self) -> None:
        first = {
            "chunk_index": 0,
            "chunk_start": 200.0,
            "chunk_end": 227.9,
            "words": [
                word(224.08, 224.34, "videolarda"),
                word(224.34, 224.88, "ko'rishguncha"),
                word(224.88, 225.72, "hammalaringizga"),
                word(225.72, 226.18, "salomatlik"),
                word(226.18, 226.50, "tilayman"),
                word(226.68, 226.84, "sog'"),
                word(226.84, 227.20, "bo'linglar"),
                word(227.26, 227.74, "charchamanglar"),
            ],
        }
        second = {
            "chunk_index": 1,
            "chunk_start": 225.0,
            "chunk_end": 227.9,
            "words": [
                word(225.00, 225.64, "malaringizga"),
                word(225.64, 226.18, "salomatlik"),
                word(226.18, 226.52, "tilayman"),
                word(226.74, 226.82, "sog'"),
                word(226.88, 227.20, "bo'linglar"),
                word(227.24, 227.72, "charchamanglar"),
            ],
        }

        chunks = remove_overlaps([first, second])
        merged = [item["word"] for chunk in chunks for item in chunk["words"]]

        self.assertIn("hammalaringizga", merged)
        self.assertNotIn("malaringizga", merged)
        self.assertEqual(merged.count("salomatlik"), 1)
        self.assertEqual(merged.count("charchamanglar"), 1)


class TranscriptCleanupTests(unittest.TestCase):
    def test_normalizes_apostrophe_spacing(self) -> None:
        self.assertEqual(normalize_spaces("to'g 'rirog'i"), "to'g'rirog'i")
        self.assertEqual(normalize_spaces("sog' bo'linglar"), "sog' bo'linglar")

    def test_polishes_formatting_without_changing_vocabulary(self) -> None:
        actual = polish_transcript_text(
            "cobit uchun sartnoma. Машинани расрочкага oling. "
            "ikki yuz'n to'rt million. 29.8% ustiga. 18 oy."
        )
        self.assertEqual(
            actual,
            "Cobit uchun sartnoma. Машинани расрочкага oling. "
            "Ikki yuz'n to'rt million. 29.8% ustiga. 18 oy.",
        )


if __name__ == "__main__":
    unittest.main()
