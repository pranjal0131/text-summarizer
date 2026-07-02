"""Frequency-based extractive summarization (Luhn-style baseline).

Sentences are scored by the normalized frequency of their content words,
averaged over sentence length so long sentences are not unfairly favored.
"""

from __future__ import annotations

from collections import Counter

from summarizer.extractive.base import ExtractiveSummarizer
from summarizer.preprocessing import tokenize_words


class FrequencySummarizer(ExtractiveSummarizer):
    """Scores sentences by mean normalized content-word frequency."""

    def score_sentences(self, sentences: list[str]) -> list[float]:
        tokenized = [tokenize_words(s) for s in sentences]

        word_freq: Counter[str] = Counter()
        for tokens in tokenized:
            word_freq.update(tokens)
        if not word_freq:
            return [0.0] * len(sentences)

        max_freq = max(word_freq.values())
        normalized = {word: count / max_freq for word, count in word_freq.items()}

        scores = []
        for tokens in tokenized:
            if not tokens:
                scores.append(0.0)
                continue
            # Mean (not sum) so score reflects density, not sentence length.
            scores.append(sum(normalized[t] for t in tokens) / len(tokens))
        return scores
