# -*- coding: utf-8 -*-
"""
Module: WordCloud
Visualizes the most frequent words in a visual WordCloud image.
Requires: pip install wordcloud
"""

from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def generate_word_cloud(text: str, top_n: int = 20):
    """Generates and displays a WordCloud of top_n words."""
    stop_words = set(stopwords.words("english"))
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    top_counts = dict(Counter(tokens).most_common(top_n))

    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(top_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - Top {top_n} Frequent Words")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample_text = """
    Food brings people together across diverse cultures. Good food nourishes the body
    and comforts the mind. Fresh food contains vital nutrients and delicious flavors.
    Cooking organic food creates healthy dining habits. Traditional food reflects regional heritage.
    """
    print("Generating Word Cloud...")
    generate_word_cloud(sample_text, top_n=15)