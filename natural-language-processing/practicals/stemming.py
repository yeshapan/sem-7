# -*- coding: utf-8 -*-
"""
Module: Stemming
Reduces words to their crude base/stem form using PorterStemmer.
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def stem_words(words: list) -> list:
    """Apply PorterStemmer to a list of tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


if __name__ == "__main__":
    sample_words = ["learning", "studies", "extracting", "generative", "empirically", "algorithms"]

    stemmed = stem_words(sample_words)
    print("--- Stemming Results ---")
    for original, stem in zip(sample_words, stemmed):
        print(f"{original:<15} -> {stem}")