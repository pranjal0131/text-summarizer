"""TF-IDF vectorization and cosine similarity, implemented from scratch.

Vectors are represented as sparse ``dict[str, float]`` mappings so that
similarity computation stays O(min(|a|, |b|)) per pair instead of O(vocab).
"""

from __future__ import annotations

import math
from collections import Counter

SparseVector = dict[str, float]


def build_tfidf_vectors(tokenized_docs: list[list[str]]) -> list[SparseVector]:
    """Compute a TF-IDF vector for each tokenized document.

    TF is term frequency normalized by document length; IDF uses the
    smoothed formulation ``log((1 + N) / (1 + df)) + 1`` so terms present
    in every document still carry a small positive weight.
    """
    num_docs = len(tokenized_docs)
    if num_docs == 0:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    idf = {
        term: math.log((1 + num_docs) / (1 + df)) + 1.0
        for term, df in doc_freq.items()
    }

    vectors: list[SparseVector] = []
    for tokens in tokenized_docs:
        if not tokens:
            vectors.append({})
            continue
        term_counts = Counter(tokens)
        total = len(tokens)
        vectors.append(
            {term: (count / total) * idf[term] for term, count in term_counts.items()}
        )
    return vectors


def cosine_similarity(vec_a: SparseVector, vec_b: SparseVector) -> float:
    """Cosine similarity between two sparse vectors, in [0, 1]."""
    if not vec_a or not vec_b:
        return 0.0
    # Iterate over the smaller vector for efficiency.
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())
    if dot == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    return dot / (norm_a * norm_b)
