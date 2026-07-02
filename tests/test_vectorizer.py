import math

import pytest

from summarizer.vectorizer import build_tfidf_vectors, cosine_similarity


class TestTfidf:
    def test_empty_corpus(self):
        assert build_tfidf_vectors([]) == []

    def test_empty_document_gets_empty_vector(self):
        vectors = build_tfidf_vectors([["word"], []])
        assert vectors[1] == {}

    def test_rare_terms_weighted_higher(self):
        docs = [["common", "rare"], ["common"], ["common"]]
        vectors = build_tfidf_vectors(docs)
        assert vectors[0]["rare"] > vectors[0]["common"]

    def test_all_terms_positive(self):
        docs = [["a", "b"], ["a", "c"]]
        for vector in build_tfidf_vectors(docs):
            assert all(weight > 0 for weight in vector.values())


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = {"a": 1.0, "b": 2.0}
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_empty_vector(self):
        assert cosine_similarity({}, {"a": 1.0}) == 0.0

    def test_known_value(self):
        # cos([1,1], [1,0]) = 1/sqrt(2)
        result = cosine_similarity({"a": 1.0, "b": 1.0}, {"a": 1.0})
        assert result == pytest.approx(1 / math.sqrt(2))
