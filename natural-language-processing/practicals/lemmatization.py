# -*- coding: utf-8 -*-
"""
Module: POS-aware Lemmatization
Converts words to their canonical dictionary base form using POS tags.
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


def get_wordnet_pos(treebank_tag: str):
    """Map POS tag to WordNet-compatible character."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_with_pos(text: str) -> list:
    """Tokenize, tag POS, and lemmatize accordingly."""
    lemmatizer = WordNetLemmatizer()
    tokens = [w for w in word_tokenize(text) if w.isalpha()]
    pos_tags = nltk.pos_tag(tokens)

    lemmatized = [
        (word, tag, lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag)))
        for word, tag in pos_tags
    ]
    return lemmatized


if __name__ == "__main__":
    sample_text = "The striped bats are flying and running faster than the best athletes."

    results = lemmatize_with_pos(sample_text)
    print(f"{'Word':<15} {'POS Tag':<10} {'Lemmatized':<15}")
    print("-" * 42)
    for word, tag, lemma in results:
        print(f"{word:<15} {tag:<10} {lemma:<15}")