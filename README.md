# Text Summarizer

Extractive and abstractive text summarization, built from first principles.
The core algorithms — **TextRank (graph-based PageRank)**, **TF-IDF
vectorization**, and **ROUGE evaluation metrics** — are implemented from
scratch with **zero runtime dependencies**, and exposed through a **REST API**,
a **CLI**, and a clean Python library interface.

[![CI](https://github.com/pranjal0131/text_summarization/actions/workflows/ci.yml/badge.svg)](../../actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Highlights

- **TextRank from scratch** — sentence similarity graph (TF-IDF + cosine)
  ranked with weighted PageRank power iteration, including damping,
  convergence detection, and dangling-node handling. No `networkx`, no
  `scikit-learn`.
- **ROUGE-1/2/L from scratch** — n-gram overlap with clipped counts, and
  ROUGE-L via longest-common-subsequence dynamic programming in
  O(min(m, n)) memory.
- **Zero-dependency core** — `pip install .` pulls in nothing; FastAPI and
  transformer extras are opt-in.
- **Production hygiene** — typed code, pytest suite, ruff linting,
  multi-version CI, multi-stage Docker image running as non-root.

## Architecture

```
                          ┌──────────────────────────────┐
  text ──► preprocessing ─► split_sentences / tokenize   │
                          └───────────────┬──────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │  TF-IDF sparse vectors        │
                          │  (vectorizer.py)              │
                          └───────────────┬──────────────┘
                                          ▼
        ┌─────────────────────────────────────────────────────┐
        │  Extractive summarizers (strategy pattern)          │
        │  ├─ TextRankSummarizer: similarity graph + PageRank │
        │  └─ FrequencySummarizer: Luhn-style baseline        │
        └───────────────────────┬─────────────────────────────┘
                                ▼
              top-k sentences, restored to document order
                                │
        ┌───────────┬───────────┴──────────┬────────────────┐
        ▼           ▼                      ▼                ▼
      CLI      FastAPI service      Python library    ROUGE evaluation
```

The abstractive path (`summarizer.abstractive.BartSummarizer`) wraps a
seq2seq transformer and lazy-loads `transformers`/`torch` only when used.

## Quick start

```bash
git clone https://github.com/pranjal0131/text_summarization.git
cd text_summarization
pip install -e ".[dev,api]"
```

### Library

```python
from summarizer import TextRankSummarizer

summarizer = TextRankSummarizer()
print(summarizer.summarize(long_text, num_sentences=3))

# Inspect per-sentence salience scores
for sentence in summarizer.rank(long_text):
    print(f"{sentence.score:.4f}  {sentence.text}")
```

### CLI

```bash
summarize --file examples/sample.txt --sentences 3
summarize --text "..." --method frequency
cat article.txt | summarize --show-scores
```

### REST API

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "<long document>", "method": "textrank", "num_sentences": 3}'
```

```json
{
  "summary": "...",
  "method": "textrank",
  "input_sentences": 12,
  "summary_sentences": 3,
  "compression_ratio": 0.31,
  "processing_time_ms": 4.2
}
```

Interactive docs at `http://localhost:8000/docs` (OpenAPI/Swagger).

### Docker

```bash
docker build -t text-summarizer .
docker run -p 8000:8000 text-summarizer
```

### Evaluation

```python
from summarizer.evaluation import rouge_n, rouge_l

score = rouge_l(candidate_summary, reference_summary)
print(score.precision, score.recall, score.f1)
```

## How TextRank works

1. **Sentence graph** — every sentence is a node; edges are weighted by the
   cosine similarity of the sentences' TF-IDF vectors. Weak edges (below a
   threshold) are pruned to de-noise the graph.
2. **PageRank** — the random-surfer model: with probability `d = 0.85`
   follow a similarity edge (proportionally to its weight), otherwise jump
   to a random sentence. Power iteration runs until the L1 change in the
   score vector drops below `1e-5`.
3. **Selection** — the top-k sentences by stationary score are returned in
   their original document order.

Complexity: `O(n² · t)` graph construction for `n` sentences of ~`t` tokens,
plus `O(k · n²)` for `k` PageRank iterations.

## Project layout

```
summarizer/
├── preprocessing.py        # sentence splitting, tokenization, stopwords
├── vectorizer.py           # TF-IDF + cosine similarity (sparse dicts)
├── extractive/
│   ├── base.py             # ExtractiveSummarizer ABC (strategy pattern)
│   ├── textrank.py         # graph + PageRank power iteration
│   └── frequency.py        # frequency baseline
├── abstractive/bart.py     # optional transformer wrapper (lazy import)
├── evaluation/rouge.py     # ROUGE-1/2/L from scratch (LCS DP)
└── cli.py                  # argparse CLI (`summarize`)
api/main.py                 # FastAPI service
tests/                      # pytest suite
notebooks/legacy/           # original Colab experiments (kept for history)
```

## Testing

```bash
pytest          # run the suite
ruff check .    # lint
```

## Roadmap

- [ ] MMR (maximal marginal relevance) re-ranking to reduce redundancy
- [ ] Multi-document summarization
- [ ] Benchmark on CNN/DailyMail with ROUGE against published baselines
- [ ] Fine-tuned BART checkpoint served behind the same API

## License

MIT
