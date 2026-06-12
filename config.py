from pathlib import Path
import re

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

# Audio is sent to Whisper in overlapping chunks. This is not the same as
# correction/display segmentation.
ASR_CHUNK_SECONDS = 30
ASR_CHUNK_OVERLAP_SECONDS = 5
ASR_CHUNK_OVERLAP_SAFE_PADDING_SECONDS = 0.75


# =============================================================================
# TRANSCRIPT CORRECTOR MODEL
# =============================================================================

CORRECTOR_MODEL_PATH = "islomov/rubai-corrector-transcript-uz"

# Use "auto" for CUDA if available, otherwise CPU.
# On Mac / CPU setup, it will automatically use CPU.
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
    "ki", "bu", "shu", "o'sha", "mana", "ya'ni", "masalan",
    "и", "а", "но", "или", "что", "если", "потому", "для", "с",
    "and", "or", "but", "if", "because", "for", "with", "to", "of",
}

PUNCTUATION_END_RE = re.compile(r"[.!?…]+[\"')\]]*$")
