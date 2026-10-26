# -*- coding: utf-8 -*-
"""
Module: POS Tagging & Filtering
Extracts nouns, verbs, and lemmatized root adjectives separately.
"""

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def extract_pos_categories(text: str) -> dict:
    """Extract all nouns, verbs, and root adjectives separately."""
    lemmatizer = WordNetLemmatizer()
    tokens = [w for w in word_tokenize(text) if w.isalpha()]
    tags = nltk.pos_tag(tokens)

    nouns = [word for word, tag in tags if tag.startswith("NN")]
    verbs = [word for word, tag in tags if tag.startswith("VB")]
    adjectives = [
        lemmatizer.lemmatize(word.lower(), wordnet.ADJ)
        for word, tag in tags if tag.startswith("JJ")
    ]

    return {
        "all_tags": tags,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives
    }


if __name__ == "__main__":
    sample_text = (
        "Machine learning models can predict future results accurately. "
        "Students build powerful neural networks using clean datasets."
    )

    data = extract_pos_categories(sample_text)
    print("Nouns:", data["nouns"])
    print("Verbs:", data["verbs"])
    print(f"Adjectives ({len(data['adjectives'])} total, root form):", data["adjectives"])