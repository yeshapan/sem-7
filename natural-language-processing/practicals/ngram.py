# -*- coding: utf-8 -*-
"""
Module: N-Gram Language Model
Implements 1-gram, 2-gram, 3-gram next-word prediction.
"""

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class NGramPredictor:
    def __init__(self, corpus: str):
        self.words = [w.lower() for w in word_tokenize(corpus) if w.isalpha()]
        self.unigrams = Counter(ngrams(self.words, 1))
        self.bigrams = Counter(ngrams(self.words, 2))
        self.trigrams = Counter(ngrams(self.words, 3))

    def predict_1gram(self) -> str:
        """Returns the most frequent single token."""
        return self.unigrams.most_common(1)[0][0][0]

    def predict_2gram(self, prev_word: str) -> str:
        """Predicts next token using bigram counts."""
        target = prev_word.lower()
        candidates = {w2: cnt for (w1, w2), cnt in self.bigrams.items() if w1 == target}
        return max(candidates, key=candidates.get) if candidates else "No prediction"

    def predict_3gram(self, prev_two_words: str) -> str:
        """Predicts next token using trigram counts."""
        parts = prev_two_words.lower().split()
        if len(parts) < 2:
            return "Requires 2 words"
        w1, w2 = parts[-2], parts[-1]
        candidates = {w3: cnt for (a, b, w3), cnt in self.trigrams.items() if a == w1 and b == w2}
        return max(candidates, key=candidates.get) if candidates else "No prediction"


if __name__ == "__main__":
    training_data = """
    machine learning is a branch of artificial intelligence
    machine learning uses data to learn patterns
    machine learning models can predict future results
    deep learning is a part of machine learning
    artificial intelligence helps computers solve problems
    data science uses machine learning for prediction
    """

    model = NGramPredictor(training_data)
    print("1-Gram Prediction:", model.predict_1gram())
    print("2-Gram Prediction for 'machine':", model.predict_2gram("machine"))
    print("3-Gram Prediction for 'machine learning':", model.predict_3gram("machine learning"))