from summarizer.preprocessing import split_sentences, tokenize_words


class TestSplitSentences:
    def test_basic_split(self):
        text = "This is one. This is two! Is this three?"
        assert split_sentences(text) == [
            "This is one.",
            "This is two!",
            "Is this three?",
        ]

    def test_glued_sentences(self):
        # Colab-style text where sentences are joined without a space.
        text = "Kindness matters.It spreads far and wide."
        assert split_sentences(text) == [
            "Kindness matters.",
            "It spreads far and wide.",
        ]

    def test_abbreviations_not_split(self):
        text = "Dr. Smith met Mr. Jones. They talked."
        assert split_sentences(text) == ["Dr. Smith met Mr. Jones.", "They talked."]

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \n\t  ") == []

    def test_whitespace_normalized(self):
        text = "First   sentence\nwith newline. Second one."
        assert split_sentences(text) == [
            "First sentence with newline.",
            "Second one.",
        ]


class TestTokenizeWords:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize_words("Hello, World!") == ["hello", "world"]

    def test_removes_stopwords(self):
        assert tokenize_words("the cat is on the mat") == ["cat", "mat"]

    def test_keeps_stopwords_when_asked(self):
        tokens = tokenize_words("the cat", remove_stopwords=False)
        assert tokens == ["the", "cat"]

    def test_hyphenated_and_apostrophe_words(self):
        tokens = tokenize_words("state-of-the-art model", remove_stopwords=False)
        assert "state-of-the-art" in tokens
