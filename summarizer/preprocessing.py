"""Text preprocessing utilities: sentence splitting, tokenization, stopwords.

Implemented with the standard library only so the core package has zero
runtime dependencies.
"""

from __future__ import annotations

import re

# Common abbreviations that end with a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "inc", "ltd", "co", "corp", "no", "vol", "fig", "al",
}

# A compact English stopword list (standard library only, no NLTK download).
STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are aren't as at
    be because been before being below between both but by can can't cannot
    could couldn't did didn't do does doesn't doing don't down during each
    few for from further had hadn't has hasn't have haven't having he he'd
    he'll he's her here here's hers herself him himself his how how's i i'd
    i'll i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our ours
    ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them themselves
    then there there's these they they'd they'll they're they've this those
    through to too under until up very was wasn't we we'd we'll we're we've
    were weren't what what's when when's where where's which while who who's
    whom why why's with won't would wouldn't you you'd you'll you're you've
    your yours yourself yourselves s t don ve ll re d m""".split()
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]*")
# Sentence boundary: terminal punctuation followed by whitespace + capital,
# or by end of string. Also handles the "word.Word" glued-sentence case.
_GLUED_SENTENCE_RE = re.compile(r"(?<=[a-z])([.!?])(?=[A-Z])")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split raw text into sentences.

    Handles glued sentences ("end.Start"), common abbreviations, and
    trailing quotes/brackets after terminal punctuation.
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    # Insert a space where two sentences are glued together without one.
    text = _GLUED_SENTENCE_RE.sub(r"\1 ", text)

    raw_parts = _SENTENCE_END_RE.split(text)
    sentences: list[str] = []
    buffer = ""
    for part in raw_parts:
        candidate = f"{buffer} {part}".strip() if buffer else part
        last_word = candidate.rstrip(".!?\"')]").rsplit(" ", 1)[-1].lower()
        # If the split point was an abbreviation, keep accumulating.
        if candidate.endswith(".") and last_word in _ABBREVIATIONS:
            buffer = candidate
        else:
            sentences.append(candidate)
            buffer = ""
    if buffer:
        sentences.append(buffer)
    return [s for s in (s.strip() for s in sentences) if s]


def tokenize_words(text: str, remove_stopwords: bool = True) -> list[str]:
    """Extract lowercase word tokens, optionally dropping stopwords."""
    tokens = [match.group(0).lower() for match in _WORD_RE.finditer(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
