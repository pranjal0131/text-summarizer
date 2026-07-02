"""Shared interface for extractive summarizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from summarizer.preprocessing import split_sentences


@dataclass(frozen=True)
class ScoredSentence:
    """A sentence with its position in the source text and salience score."""

    index: int
    text: str
    score: float


class ExtractiveSummarizer(ABC):
    """Base class implementing the select-and-reorder summarization flow.

    Subclasses only implement :meth:`score_sentences`; sentence selection
    and restoring the original document order are handled here.
    """

    @abstractmethod
    def score_sentences(self, sentences: list[str]) -> list[float]:
        """Return one salience score per sentence."""

    def rank(self, text: str) -> list[ScoredSentence]:
        """Score every sentence in ``text``, sorted by score descending."""
        sentences = split_sentences(text)
        scores = self.score_sentences(sentences)
        ranked = [
            ScoredSentence(index=i, text=s, score=score)
            for i, (s, score) in enumerate(zip(sentences, scores))
        ]
        return sorted(ranked, key=lambda s: s.score, reverse=True)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Return the top ``num_sentences`` sentences in original order."""
        if num_sentences < 1:
            raise ValueError("num_sentences must be >= 1")
        top = self.rank(text)[:num_sentences]
        # Present selected sentences in document order for readability.
        top.sort(key=lambda s: s.index)
        return " ".join(s.text for s in top)
