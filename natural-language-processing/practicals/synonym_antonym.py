# -*- coding: utf-8 -*-
"""
Module: Synonyms and Antonyms Lookup
Searches if candidate words exist in the corpus and returns their WordNet synonyms/antonyms.
"""

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def find_word_relations(target_words: list, corpus_text: str) -> dict:
    """Checks if each target word is present in corpus; extracts synonyms/antonyms."""
    corpus_tokens = set([w.lower() for w in word_tokenize(corpus_text) if w.isalpha()])
    results = {}

    for word in target_words:
        w_lower = word.lower()
        if w_lower not in corpus_tokens:
            results[word] = {"status": "out of corpus"}
            continue

        synonyms = set()
        antonyms = set()
        for synset in wordnet.synsets(w_lower):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name().replace('_', ' '))

        results[word] = {
            "status": "found",
            "synonyms": list(synonyms)[:5],
            "antonyms": list(antonyms)[:5] if antonyms else ["Not Found"]
        }

    return results


if __name__ == "__main__":
    corpus = "Students learn machine learning to predict the future results using clean data."
    query_words = ["learning", "future", "data", "blockchain"]

    output = find_word_relations(query_words, corpus)
    for word, info in output.items():
        if info["status"] == "out of corpus":
            print(f"'{word}' -> out of corpus")
        else:
            print(f"\nWord: '{word}'")
            print(f"  Synonyms : {info['synonyms']}")
            print(f"  Antonyms : {info['antonyms']}")