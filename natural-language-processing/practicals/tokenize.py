# -*- coding: utf-8 -*-
"""
Module: Tokenization
Splits text into individual sentences and words.
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def tokenize_sentences(text: str) -> list:
    """Split text into sentences."""
    return sent_tokenize(text)


def tokenize_words(text: str) -> list:
    """Split text into word tokens."""
    return word_tokenize(text)


if __name__ == "__main__":
    sample_text = (
        "Natural Language Processing is a subfield of computer science. "
        "It focuses on the interactions between human language and computers. "
        "Are you excited to learn NLP?"
    )

    print("--- Sentence Tokenization ---")
    sentences = tokenize_sentences(sample_text)
    for idx, sentence in enumerate(sentences, 1):
        print(f"{idx}: {sentence}")

    print("\n--- Word Tokenization ---")
    words = tokenize_words(sample_text)
    print(words)