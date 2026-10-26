"""
Module: Stopword Removal
Filters out punctuation, non-alphabetic tokens, and standard stopwords.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def remove_stopwords(text: str, language: str = "english") -> list:
    """Tokenize and filter out non-alphabetic tokens and language stopwords."""
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(text)
    return [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]


if __name__ == "__main__":
    sample_text = (
        "This is an empirical study designed to excavate significant term "
        "extractions using Natural Language Processing tools and Python."
    )

    cleaned_tokens = remove_stopwords(sample_text)
    print("--- Cleaned Tokens (Stopwords Removed) ---")
    print(cleaned_tokens)