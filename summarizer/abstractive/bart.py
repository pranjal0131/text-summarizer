"""Abstractive summarization backed by a seq2seq transformer.

The heavy `transformers`/`torch` stack is imported lazily so the core
package stays dependency-free. Install with ``pip install .[ml]``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Abstractive summarization requires the optional ML dependencies. "
    "Install them with: pip install 'text-summarizer[ml]' "
    "(or: pip install transformers torch)"
)


class BartSummarizer:
    """Wraps a Hugging Face summarization pipeline with lazy model loading.

    The model is only downloaded/loaded on the first ``summarize`` call,
    so constructing this class is cheap and safe in web-server startup.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn") -> None:
        self.model_name = model_name
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(_INSTALL_HINT) from exc
            logger.info("Loading summarization model %s", self.model_name)
            self._pipeline = pipeline("summarization", model=self.model_name)
        return self._pipeline

    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
    ) -> str:
        """Generate an abstractive summary of ``text``."""
        if not text.strip():
            raise ValueError("text must not be empty")
        if min_length >= max_length:
            raise ValueError("min_length must be < max_length")
        pipe = self._load()
        result = pipe(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()
