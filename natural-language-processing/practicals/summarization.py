# -*- coding: utf-8 -*-
"""
Module: Extractive Summarizer
Scores sentences based on word frequencies and compresses text down to ~1/10th of original words.
"""

from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def summarize_text(text: str, ratio: float = 0.1) -> str:
    """Extracts highest-scoring sentences up to a ratio-th of original word count."""
    sentences = sent_tokenize(text)
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    target_words = max(10, int(len(words) * ratio))

    stop_words = set(stopwords.words("english"))
    freq_table = Counter([w for w in words if w not in stop_words])

    # Score each sentence by its content words normalized by length
    sentence_scores = {}
    for sent in sentences:
        sent_words = [w.lower() for w in word_tokenize(sent) if w.isalpha()]
        if not sent_words:
            continue
        score = sum(freq_table.get(w, 0) for w in sent_words) / len(sent_words)
        sentence_scores[sent] = score

    # Select top sentences
    ranked_sentences = sorted(sentence_scores.keys(), key=lambda s: sentence_scores[s], reverse=True)
    summary_sentences = []
    collected_word_count = 0

    for sent in ranked_sentences:
        summary_sentences.append(sent)
        collected_word_count += len(word_tokenize(sent))
        if collected_word_count >= target_words:
            break

    # Restore chronological order of sentences
    summary_sentences.sort(key=lambda s: sentences.index(s))
    return " ".join(summary_sentences)


if __name__ == "__main__":
    long_text = """
    Keyword extraction is the task of identifying important terms or phrases that are most representative.
    Although the process of automatic extraction from titles is an old method, it was mainly for single documents.
    Our approach differs from previous research works on keyword extraction in several aspects.
    For those who are non-experts of the scientific fields, understanding scientific research trends is difficult.
    The purpose of this study is to develop an automatic method of obtaining overviews of scientific fields.
    This empirical study excavates significant term extraction using Natural Language Processing tools.
    More than 15000 titles saved in a file were used to compare significant terms.
    A light-weight unsupervised title extractor was used to extract the key results.
    Based on our analysis, it can be concluded that these algorithms can be used for other fields too.
    Our algorithm could be a solution to reduce the labour-intensive manual indexing process.
    """

    summary = summarize_text(long_text, ratio=0.2)
    print("--- Extractive Summary ---")
    print(summary)
    print(f"\nOriginal words: {len(word_tokenize(long_text))} | Summary words: {len(word_tokenize(summary))}")