# -*- coding: utf-8 -*-
"""
Module: Word Frequency & Visualization
Calculates word frequencies excluding stopwords and plots a bar chart.
"""

from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def get_word_frequencies(text: str, top_n: int = 15) -> list:
    """Extracts top_n frequent non-stopword tokens."""
    stop_words = set(stopwords.words("english"))
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    return Counter(tokens).most_common(top_n)


def plot_word_frequency(frequencies: list):
    """Renders a matplotlib bar chart."""
    if not frequencies:
        print("No frequencies to plot.")
        return
    words, counts = zip(*frequencies)

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Most Frequent Words (Excluding Stopwords)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample_document = """
    Keyword extraction is the task of identifying important terms or phrases.
    Automatic extraction of keywords from titles is an old extraction method.
    Significant terms of scientific titles are extracted using natural language processing tools.
    Unsupervised extraction tools extract significant keywords for non-experts.
    """

    freq_list = get_word_frequencies(sample_document, top_n=10)
    print("--- Top 10 Frequent Words ---")
    for word, cnt in freq_list:
        print(f"{word:<15}: {cnt}")

    # Uncomment below to display the GUI plot
    # plot_word_frequency(freq_list)