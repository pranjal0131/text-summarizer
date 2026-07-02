"""ROUGE metrics implemented from scratch.

ROUGE-N counts n-gram overlap between a candidate summary and a reference;
ROUGE-L uses the longest common subsequence (classic O(m*n) dynamic
programming) so it rewards in-order matches without requiring contiguity.

Reference: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries"
(ACL 2004 Workshop).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from summarizer.preprocessing import tokenize_words


@dataclass(frozen=True)
class RougeScore:
    """Precision / recall / F1 triple for one ROUGE variant."""

    precision: float
    recall: float
    f1: float


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def rouge_n(candidate: str, reference: str, n: int = 1) -> RougeScore:
    """ROUGE-N: clipped n-gram overlap between candidate and reference."""
    if n < 1:
        raise ValueError("n must be >= 1")
    cand_tokens = tokenize_words(candidate, remove_stopwords=False)
    ref_tokens = tokenize_words(reference, remove_stopwords=False)

    cand_ngrams = _ngrams(cand_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    if not cand_ngrams or not ref_ngrams:
        return RougeScore(0.0, 0.0, 0.0)

    overlap = sum((cand_ngrams & ref_ngrams).values())
    precision = overlap / sum(cand_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    return RougeScore(precision, recall, _f1(precision, recall))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length via DP with O(min(m,n)) memory."""
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> RougeScore:
    """ROUGE-L: longest-common-subsequence based precision/recall/F1."""
    cand_tokens = tokenize_words(candidate, remove_stopwords=False)
    ref_tokens = tokenize_words(reference, remove_stopwords=False)
    if not cand_tokens or not ref_tokens:
        return RougeScore(0.0, 0.0, 0.0)

    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    return RougeScore(precision, recall, _f1(precision, recall))
