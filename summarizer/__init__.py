"""Text summarization toolkit.

Provides extractive summarization (TextRank, frequency-based) implemented
from scratch with zero runtime dependencies, an optional abstractive
summarizer backed by transformer models, and ROUGE evaluation metrics.
"""

from summarizer.extractive.frequency import FrequencySummarizer
from summarizer.extractive.textrank import TextRankSummarizer

__version__ = "1.0.0"

__all__ = [
    "TextRankSummarizer",
    "FrequencySummarizer",
    "__version__",
]
