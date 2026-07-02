import pytest

from summarizer.extractive import FrequencySummarizer, TextRankSummarizer
from summarizer.preprocessing import split_sentences

SAMPLE_TEXT = (
    "Machine learning is a field of artificial intelligence. "
    "Machine learning systems learn patterns from data. "
    "Deep learning is a subset of machine learning using neural networks. "
    "Neural networks are inspired by the human brain. "
    "The weather today is sunny and warm. "
    "Data is the fuel that powers machine learning models."
)

ALL_SUMMARIZERS = [TextRankSummarizer, FrequencySummarizer]


@pytest.mark.parametrize("summarizer_cls", ALL_SUMMARIZERS)
class TestSummarizerContract:
    """Behavioral contract every extractive summarizer must satisfy."""

    def test_returns_requested_sentence_count(self, summarizer_cls):
        summary = summarizer_cls().summarize(SAMPLE_TEXT, num_sentences=2)
        assert len(split_sentences(summary)) == 2

    def test_summary_sentences_come_from_source(self, summarizer_cls):
        source = split_sentences(SAMPLE_TEXT)
        summary = summarizer_cls().summarize(SAMPLE_TEXT, num_sentences=3)
        for sentence in split_sentences(summary):
            assert sentence in source

    def test_preserves_original_order(self, summarizer_cls):
        source = split_sentences(SAMPLE_TEXT)
        summary = split_sentences(
            summarizer_cls().summarize(SAMPLE_TEXT, num_sentences=3)
        )
        positions = [source.index(s) for s in summary]
        assert positions == sorted(positions)

    def test_num_sentences_larger_than_text(self, summarizer_cls):
        text = "One sentence. Two sentences."
        summary = summarizer_cls().summarize(text, num_sentences=10)
        assert len(split_sentences(summary)) == 2

    def test_invalid_num_sentences_raises(self, summarizer_cls):
        with pytest.raises(ValueError):
            summarizer_cls().summarize(SAMPLE_TEXT, num_sentences=0)

    def test_scores_align_with_sentences(self, summarizer_cls):
        sentences = split_sentences(SAMPLE_TEXT)
        scores = summarizer_cls().score_sentences(sentences)
        assert len(scores) == len(sentences)

    def test_empty_text(self, summarizer_cls):
        assert summarizer_cls().summarize("", num_sentences=3) == ""


class TestTextRank:
    def test_scores_form_probability_distribution(self):
        sentences = split_sentences(SAMPLE_TEXT)
        scores = TextRankSummarizer().score_sentences(sentences)
        assert sum(scores) == pytest.approx(1.0, abs=1e-3)
        assert all(score > 0 for score in scores)

    def test_off_topic_sentence_ranked_low(self):
        # "The weather..." shares no vocabulary with the ML-themed corpus,
        # so PageRank should give it a low stationary score.
        sentences = split_sentences(SAMPLE_TEXT)
        scores = TextRankSummarizer().score_sentences(sentences)
        weather_idx = next(i for i, s in enumerate(sentences) if "weather" in s)
        assert scores[weather_idx] == min(scores)

    def test_single_sentence(self):
        assert TextRankSummarizer().score_sentences(["Only one."]) == [1.0]

    def test_invalid_damping_rejected(self):
        with pytest.raises(ValueError):
            TextRankSummarizer(damping=1.5)


class TestFrequency:
    def test_stopword_only_sentence_scores_zero(self):
        scores = FrequencySummarizer().score_sentences(
            ["It is what it is.", "Machine learning models learn."]
        )
        assert scores[0] == 0.0
        assert scores[1] > 0.0
