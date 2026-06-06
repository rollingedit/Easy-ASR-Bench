from __future__ import annotations

import re

from jiwer import cer, wer


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def compare_texts(left: str, right: str, reference: str | None = None) -> dict[str, float]:
    char_distance = levenshtein(left, right)
    left_words = left.split()
    right_words = right.split()
    word_distance = levenshtein(left_words, right_words)
    clean_left = normalize_text(left)
    clean_right = normalize_text(right)
    metrics = {
        "character_edit_distance": float(char_distance),
        "normalized_character_edit_distance": char_distance / max(1, len(right)),
        "word_edit_distance": float(word_distance),
        "normalized_word_edit_distance": word_distance / max(1, len(right_words)),
        "case_punctuation_insensitive_difference": cer(clean_right, clean_left) if clean_right or clean_left else 0.0,
    }
    if reference is not None:
        metrics["left_wer_against_reference"] = wer(reference, left)
        metrics["right_wer_against_reference"] = wer(reference, right)
        metrics["left_cer_against_reference"] = cer(reference, left)
        metrics["right_cer_against_reference"] = cer(reference, right)
    return metrics
