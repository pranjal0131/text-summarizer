"""TextRank extractive summarization, implemented from scratch.

Reference: Mihalcea & Tarau, "TextRank: Bringing Order into Text" (EMNLP 2004).

Pipeline:
    1. Split the document into sentences and tokenize each one.
    2. Build TF-IDF vectors and a weighted similarity graph where nodes are
       sentences and edge weights are cosine similarities.
    3. Run PageRank (power iteration with damping) over the graph; the
       stationary score of each node is its salience.

Complexity: O(n^2 * t) to build the graph for n sentences with ~t tokens
each, and O(k * n^2) for k power-iteration steps.
"""

from __future__ import annotations

import logging

from summarizer.extractive.base import ExtractiveSummarizer
from summarizer.preprocessing import tokenize_words
from summarizer.vectorizer import build_tfidf_vectors, cosine_similarity

logger = logging.getLogger(__name__)


class TextRankSummarizer(ExtractiveSummarizer):
    """Graph-based summarizer ranking sentences with PageRank.

    Args:
        damping: PageRank damping factor (probability of following an edge
            rather than teleporting). 0.85 is the standard choice.
        convergence_threshold: stop when the L1 change in scores between
            iterations falls below this value.
        max_iterations: hard cap on power-iteration steps.
        similarity_threshold: edges weaker than this are dropped, which
            de-noises the graph and speeds up convergence.
    """

    def __init__(
        self,
        damping: float = 0.85,
        convergence_threshold: float = 1e-5,
        max_iterations: int = 100,
        similarity_threshold: float = 0.05,
    ) -> None:
        if not 0.0 < damping < 1.0:
            raise ValueError("damping must be in (0, 1)")
        self.damping = damping
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold

    def score_sentences(self, sentences: list[str]) -> list[float]:
        n = len(sentences)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        matrix = self._build_similarity_matrix(sentences)
        return self._pagerank(matrix)

    def _build_similarity_matrix(self, sentences: list[str]) -> list[list[float]]:
        tokenized = [tokenize_words(s) for s in sentences]
        vectors = build_tfidf_vectors(tokenized)
        n = len(sentences)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                weight = cosine_similarity(vectors[i], vectors[j])
                if weight >= self.similarity_threshold:
                    matrix[i][j] = weight
                    matrix[j][i] = weight
        return matrix

    def _pagerank(self, matrix: list[list[float]]) -> list[float]:
        """Weighted PageRank via power iteration.

        Dangling nodes (sentences with no edges above the threshold)
        distribute their rank uniformly, which keeps the score vector a
        proper probability distribution.
        """
        n = len(matrix)
        out_weight = [sum(row) for row in matrix]
        scores = [1.0 / n] * n
        teleport = (1.0 - self.damping) / n

        for iteration in range(self.max_iterations):
            dangling_mass = sum(
                scores[j] for j in range(n) if out_weight[j] == 0.0
            )
            new_scores = []
            for i in range(n):
                rank_from_edges = sum(
                    (matrix[j][i] / out_weight[j]) * scores[j]
                    for j in range(n)
                    if matrix[j][i] > 0.0
                )
                new_scores.append(
                    teleport
                    + self.damping * (rank_from_edges + dangling_mass / n)
                )
            delta = sum(abs(new - old) for new, old in zip(new_scores, scores))
            scores = new_scores
            if delta < self.convergence_threshold:
                logger.debug("PageRank converged after %d iterations", iteration + 1)
                break
        else:
            logger.debug(
                "PageRank hit max_iterations=%d without converging",
                self.max_iterations,
            )
        return scores
