# -*- coding: utf-8 -*-
"""
Module: Keyword Extraction & Evaluation
Extracts candidate keywords and evaluates them against ground-truth keywords.
"""

from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def extract_keywords(abstract_text: str, top_n: int = 10) -> list:
    """Extracts top_n unigrams excluding stopwords."""
    stop_words = set(stopwords.words("english"))
    words = [w.lower() for w in word_tokenize(abstract_text) if w.isalpha() and w.lower() not in stop_words]
    return [word for word, _ in Counter(words).most_common(top_n)]


def evaluate_keywords(extracted_keywords: list, ground_truth: list) -> dict:
    """Compares ground-truth keywords with extracted tokens."""
    extracted_set = set([k.lower() for k in extracted_keywords])
    matched = []
    unmatched = []

    for gt in ground_truth:
        tokens = gt.lower().split()
        if any(token in extracted_set for token in tokens):
            matched.append(gt)
        else:
            unmatched.append(gt)

    return {"matched": matched, "unmatched": unmatched}


if __name__ == "__main__":
    sample_abstract = """
    Keyword extraction is the task of identifying important terms or phrases that
    are most representative of the source document. A light-weight unsupervised
    extractor was used to extract the results. Significant terms can be excavated
    using Natural Language Processing tools and python algorithms.
    """

    actual = ["keyword extraction", "natural language processing", "deep neural net", "algorithms"]

    extracted = extract_keywords(sample_abstract, top_n=6)
    evaluation = evaluate_keywords(extracted, actual)

    print("Extracted Keywords:", extracted)
    print("Matched Actual Keywords:", evaluation["matched"])
    print("Unmatched Actual Keywords:", evaluation["unmatched"])