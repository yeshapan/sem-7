# -*- coding: utf-8 -*-
"""
Module: LSTM Word Generation
Builds, trains, and generates sequential words with a minimal Keras LSTM network.
"""

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


class WordLevelLSTM:
    def __init__(self, sequence_len: int = 5):
        self.sequence_len = sequence_len
        self.tokenizer = Tokenizer()
        self.model = None

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def train(self, corpus: str, epochs: int = 60, batch_size: int = 16):
        cleaned = self.clean_text(corpus)
        self.tokenizer.fit_on_texts([cleaned])
        vocab_size = len(self.tokenizer.word_index) + 1

        tokens = self.tokenizer.texts_to_sequences([cleaned])[0]
        sequences = []
        for i in range(1, len(tokens)):
            start = max(0, i - self.sequence_len)
            sequences.append(tokens[start:i + 1])

        max_len = max(len(s) for s in sequences)
        padded = np.array(pad_sequences(sequences, maxlen=max_len, padding='pre'))
        X = padded[:, :-1]
        y = to_categorical(padded[:, -1], num_classes=vocab_size)

        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=32, input_length=X.shape[1]),
            LSTM(64),
            Dense(vocab_size, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict_next_words(self, seed: str, num_words: int = 5) -> str:
        current = seed.strip()
        idx_to_word = {idx: word for word, idx in self.tokenizer.word_index.items()}
        seq_len = self.model.input_shape[1]

        for _ in range(num_words):
            seq = self.tokenizer.texts_to_sequences([current.lower()])[0]
            padded = pad_sequences([seq], maxlen=seq_len, padding='pre')
            pred_idx = np.argmax(self.model.predict(padded, verbose=0), axis=-1)[0]
            next_word = idx_to_word.get(pred_idx, "")
            if not next_word:
                break
            current += " " + next_word

        return current


if __name__ == "__main__":
    training_corpus = """
    Food brings people together across diverse cultures. Good food nourishes the body and mind.
    Fresh food contains vital nutrients and delicious flavors. Cooking organic food creates healthy dining habits.
    Nutritious food supports energy and lifelong wellness.
    """

    lstm = WordLevelLSTM(sequence_len=4)
    print("Training LSTM model...")
    lstm.train(training_corpus, epochs=70)

    generated = lstm.predict_next_words("food", num_words=5)
    print("\n--- Next 5 Words Prediction ---")
    print(f"Seed  : 'food'")
    print(f"Output: '{generated}'")