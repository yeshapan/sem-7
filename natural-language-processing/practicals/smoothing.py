# -*- coding: utf-8 -*-
"""
Module: Language Model Smoothing
Implements Bigram Laplace Add-One and NLTK Witten-Bell Interpolated smoothing.
"""

from collections import Counter
import nltk
from nltk.lm import WittenBellInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class SmoothedLanguageModel:
    def __init__(self, corpus: str):
        self.tokens = [w.lower() for w in word_tokenize(corpus) if w.isalpha()]
        self.vocab = set(self.tokens)
        self.unigram_counts = Counter(ngrams(self.tokens, 1))
        self.bigram_counts = Counter(ngrams(self.tokens, 2))

        # Train Witten-Bell Interpolated Model
        train_data, _ = padded_everygram_pipeline(2, [self.tokens])
        self.wb_model = WittenBellInterpolated(2)
        self.wb_model.fit(train_data, vocabulary_text=self.tokens)

    def predict_laplace(self, prev_word: str):
        """Finds next token maximizing P(w2|w1) = (Count(w1, w2) + 1) / (Count(w1) + V)."""
        w1 = prev_word.lower()
        w1_count = self.unigram_counts.get((w1,), 0)
        v_size = len(self.vocab)

        best_word, highest_prob = None, -1.0
        for w2 in self.vocab:
            bg_count = self.bigram_counts.get((w1, w2), 0)
            prob = (bg_count + 1) / (w1_count + v_size)
            if prob > highest_prob:
                highest_prob = prob
                best_word = w2

        return best_word, highest_prob

    def predict_witten_bell(self, prev_word: str):
        """Predicts next token using Witten-Bell scored probability."""
        w1 = prev_word.lower()
        best_word, highest_score = None, -1.0
        for w2 in self.vocab:
            score = self.wb_model.score(w2, [w1])
            if score > highest_score:
                highest_score = score
                best_word = w2

        return best_word, highest_score


if __name__ == "__main__":
    text_corpus = """
    machine learning uses data to learn patterns
    machine learning models can predict future results
    data science uses machine learning for prediction
    """

    smoothed = SmoothedLanguageModel(text_corpus)
    laplace_pred, laplace_prob = smoothed.predict_laplace("machine")
    wb_pred, wb_prob = smoothed.predict_witten_bell("machine")

    print(f"Laplace Add-One   -> Next: '{laplace_pred}', Prob: {laplace_prob:.4f}")
    print(f"Witten-Bell Model -> Next: '{wb_pred}', Score: {wb_prob:.4f}")