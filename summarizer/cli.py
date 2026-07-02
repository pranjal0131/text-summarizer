"""Command-line interface for the summarizer.

Examples:
    summarize --file article.txt --sentences 3
    summarize --text "..." --method frequency
    cat article.txt | summarize --method textrank --show-scores
"""

from __future__ import annotations

import argparse
import logging
import sys

from summarizer import __version__
from summarizer.extractive import (
    ExtractiveSummarizer,
    FrequencySummarizer,
    TextRankSummarizer,
)

EXTRACTIVE_METHODS: dict[str, type[ExtractiveSummarizer]] = {
    "textrank": TextRankSummarizer,
    "frequency": FrequencySummarizer,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize",
        description="Summarize text using extractive or abstractive methods.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to summarize.")
    source.add_argument("--file", help="Path to a UTF-8 text file to summarize.")
    parser.add_argument(
        "--method",
        choices=[*EXTRACTIVE_METHODS, "abstractive"],
        default="textrank",
        help="Summarization algorithm (default: textrank).",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=3,
        help="Number of sentences in an extractive summary (default: 3).",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Print per-sentence salience scores instead of the summary.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, encoding="utf-8") as handle:
            return handle.read()
    if sys.stdin.isatty():
        raise SystemExit(
            "No input provided. Use --text, --file, or pipe text via stdin."
        )
    return sys.stdin.read()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    text = _read_input(args).strip()
    if not text:
        print("Error: input text is empty.", file=sys.stderr)
        return 1

    if args.method == "abstractive":
        from summarizer.abstractive import BartSummarizer

        try:
            print(BartSummarizer().summarize(text))
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    summarizer = EXTRACTIVE_METHODS[args.method]()
    if args.show_scores:
        for sentence in summarizer.rank(text):
            print(f"[{sentence.score:.4f}] ({sentence.index}) {sentence.text}")
    else:
        print(summarizer.summarize(text, num_sentences=args.sentences))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
