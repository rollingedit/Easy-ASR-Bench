from __future__ import annotations

import re
import unicodedata


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text.casefold())
    text = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'").replace("`", "'").replace("â€™", "'")
    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            chars.append(" ")
        elif category[0] in {"L", "N", "M"}:
            chars.append(char)
        elif char == "'":
            chars.append(char)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip().split()


def strict_words(text: str) -> list[str]:
    return re.sub(r"\s+", " ", text).strip().split()


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if left == right else 1),
                )
            )
        previous = current
    return previous[-1]


def align_words(reference: list[str], hypothesis: list[str]) -> list[dict]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    op = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
        op[i][0] = "delete"
    for j in range(1, cols):
        dp[0][j] = j
        op[0][j] = "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                choices = [(dp[i - 1][j - 1], "equal")]
            else:
                choices = [(dp[i - 1][j - 1] + 1, "replace")]
            choices.extend([(dp[i - 1][j] + 1, "delete"), (dp[i][j - 1] + 1, "insert")])
            dp[i][j], op[i][j] = min(choices, key=lambda item: item[0])
    i = len(reference)
    j = len(hypothesis)
    aligned: list[dict] = []
    while i > 0 or j > 0:
        action = op[i][j]
        if action == "equal":
            aligned.append({"op": "equal", "reference": reference[i - 1], "hypothesis": hypothesis[j - 1]})
            i -= 1
            j -= 1
        elif action == "replace":
            aligned.append({"op": "replace", "reference": reference[i - 1], "hypothesis": hypothesis[j - 1]})
            i -= 1
            j -= 1
        elif action == "delete":
            aligned.append({"op": "delete", "reference": reference[i - 1], "hypothesis": ""})
            i -= 1
        else:
            aligned.append({"op": "insert", "reference": "", "hypothesis": hypothesis[j - 1]})
            j -= 1
    return list(reversed(aligned))


def wer(reference: str, hypothesis: str, normalized: bool = True) -> float:
    ref = normalize_words(reference) if normalized else strict_words(reference)
    hyp = normalize_words(hypothesis) if normalized else strict_words(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def cer(reference: str, hypothesis: str, normalized: bool = False) -> float:
    if normalized:
        reference = " ".join(normalize_words(reference))
        hypothesis = " ".join(normalize_words(hypothesis))
    return edit_distance(reference, hypothesis) / max(1, len(reference))


def pairwise_metrics(left: str, right: str) -> dict:
    return {
        "normalized_wer_like_difference": wer(left, right, normalized=True),
        "strict_wer_like_difference": wer(left, right, normalized=False),
        "cer_difference": cer(left, right, normalized=False),
        "normalized_cer_difference": cer(left, right, normalized=True),
    }


def score_against_reference(reference: str, hypothesis: str) -> dict:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    alignment = align_words(ref_words, hyp_words)
    substitutions = sum(1 for item in alignment if item["op"] == "replace")
    insertions = sum(1 for item in alignment if item["op"] == "insert")
    deletions = sum(1 for item in alignment if item["op"] == "delete")
    return {
        "normalized_wer": wer(reference, hypothesis, normalized=True),
        "strict_wer": wer(reference, hypothesis, normalized=False),
        "cer": cer(reference, hypothesis, normalized=False),
        "normalized_cer": cer(reference, hypothesis, normalized=True),
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "alignment": alignment,
    }


def balanced_score(quality: float, speed_percentile: float, memory_percentile_inverse: float) -> float:
    return max(0.0, min(1.0, 0.70 * quality + 0.20 * speed_percentile + 0.10 * memory_percentile_inverse))
