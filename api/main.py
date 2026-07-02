"""FastAPI service exposing the summarizer over HTTP.

Run locally:
    uvicorn api.main:app --reload

Endpoints:
    GET  /health     - liveness probe
    POST /summarize  - summarize a document
"""

from __future__ import annotations

import logging
import time
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from summarizer import __version__
from summarizer.extractive import FrequencySummarizer, TextRankSummarizer
from summarizer.preprocessing import split_sentences

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Summarization API",
    description=(
        "Extractive summarization with TextRank (graph-based PageRank) "
        "and a frequency baseline, implemented from scratch."
    ),
    version=__version__,
)

# Summarizers are stateless, so single shared instances are thread-safe.
_SUMMARIZERS = {
    "textrank": TextRankSummarizer(),
    "frequency": FrequencySummarizer(),
}


class Method(str, Enum):
    TEXTRANK = "textrank"
    FREQUENCY = "frequency"


class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Document to summarize.",
        examples=["Long article text goes here..."],
    )
    method: Method = Field(
        default=Method.TEXTRANK, description="Summarization algorithm."
    )
    num_sentences: int = Field(
        default=3, ge=1, le=50, description="Sentences to keep in the summary."
    )


class SummarizeResponse(BaseModel):
    summary: str
    method: Method
    input_sentences: int
    summary_sentences: int
    compression_ratio: float = Field(
        description="len(summary) / len(input), lower means more compressed."
    )
    processing_time_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    sentences = split_sentences(request.text)
    if not sentences:
        raise HTTPException(
            status_code=422, detail="No sentences could be extracted from the text."
        )

    start = time.perf_counter()
    num_sentences = min(request.num_sentences, len(sentences))
    summary = _SUMMARIZERS[request.method.value].summarize(
        request.text, num_sentences=num_sentences
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "summarized %d sentences -> %d via %s in %.1fms",
        len(sentences), num_sentences, request.method.value, elapsed_ms,
    )
    return SummarizeResponse(
        summary=summary,
        method=request.method,
        input_sentences=len(sentences),
        summary_sentences=num_sentences,
        compression_ratio=round(len(summary) / len(request.text), 4),
        processing_time_ms=round(elapsed_ms, 2),
    )
