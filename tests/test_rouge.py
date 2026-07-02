import pytest

from summarizer.evaluation import rouge_l, rouge_n


class TestRougeN:
    def test_identical_texts_score_one(self):
        score = rouge_n("the cat sat on the mat", "the cat sat on the mat", n=1)
        assert score.precision == score.recall == score.f1 == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self):
        score = rouge_n("alpha beta gamma", "delta epsilon zeta", n=1)
        assert score.f1 == 0.0

    def test_known_unigram_values(self):
        # candidate: [the, cat, sat] reference: [the, cat, slept, here]
        # overlap = {the, cat} = 2 -> P = 2/3, R = 2/4
        score = rouge_n("the cat sat", "the cat slept here", n=1)
        assert score.precision == pytest.approx(2 / 3)
        assert score.recall == pytest.approx(2 / 4)

    def test_bigram_overlap(self):
        # candidate bigrams: (the,cat), (cat,sat); reference contains (the,cat)
        score = rouge_n("the cat sat", "the cat slept", n=2)
        assert score.precision == pytest.approx(1 / 2)

    def test_clipping_repeated_ngrams(self):
        # "the the the" must not get credit for "the" more times than the
        # reference contains it (clipped counts).
        score = rouge_n("the the the", "the cat", n=1)
        assert score.precision == pytest.approx(1 / 3)

    def test_empty_candidate(self):
        assert rouge_n("", "reference text", n=1).f1 == 0.0

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            rouge_n("a", "a", n=0)


class TestRougeL:
    def test_identical_texts(self):
        assert rouge_l("a b c", "a b c").f1 == pytest.approx(1.0)

    def test_subsequence_not_substring(self):
        # LCS of [a x b y c] and [a b c] is [a b c] (non-contiguous).
        score = rouge_l("a x b y c", "a b c")
        assert score.recall == pytest.approx(1.0)
        assert score.precision == pytest.approx(3 / 5)

    def test_order_matters(self):
        # Reversed tokens: LCS length 1 -> low score.
        score = rouge_l("c b a", "a b c")
        assert score.f1 == pytest.approx(1 / 3)

    def test_empty_inputs(self):
        assert rouge_l("", "").f1 == 0.0
