'''
1. Clean and tokenize the downloaded text into word sequences
2. Write a function generate_text(seed_text, num_words, model) that accepts a user-provided seed word or seed phrase (eg: "The hare" or "will you help me") 
and generates 5 number of subsequent words to form a short story or poem
'''

import re
import random
from collections import defaultdict

text = ''' Something terrible happened one day. In the forest, there was a pack of hounds hunting
hares. The hare was terrified and in serious need of help, so she raced to her friends. She went straight to the horse. "Since you're quick and have a lot of room on your back, please transport me to a safe location." The horse, on the other hand, refused. He stated, " have other essential work."
She then rushed to the buffalo. She pleaded, "Please assist me." "I would, but I already have plans." He went on to say that the goat might be of help. As a result, the hare ran over to the goat. "Please, you must assist me," she cried. "I am little and my back is weird," the goat explained. You could injure yourself."
The hare then ran over to the ram. "Will you help me?" she asked. He said, "I would want to decline."
"You know, I don't like to meddle in other people's affairs. In case you didn't know, hounds have been known to eat sheep. So, ask for help another time." You might also enjoy The Hare's Liver.
Her final chance was the calf, but he, too, turned her down. "I'm not interested in taking on too many responsibilities." As a result, I am unable to help you. The hounds were approaching. "How should I proceed?" She bolted before she could think of anything else. She dashed away with all her strength!
Luckily, she was able to run away, although she never made many friends after that. Her lesson had been learned. '''

def clean_text(raw: str) -> str:
    raw = raw.lower() # convert to lowercase
    raw = re.sub(r"[^a-z\s]", "", raw)     # keep only a-z and whitespace
    raw = re.sub(r"\s+", " ", raw).strip() # collapse whitespace
    return raw
def tokenize(clean: str) -> list:
    return clean.split()  # split into word tokens


# Run the clean + tokenize part
cleaned_text = clean_text(text)
tokens = tokenize(cleaned_text)

print("Cleaned + Tokenized Word Sequence")
print(f"Total tokens  : {len(tokens)}")
print(f"First 20 words: {tokens[:20]}")
print(f"Unique words  : {len(set(tokens))}")
print()

# Build the n-gram model
def build_ngram_model(token_list: list, n: int = 2) -> dict:
    model = defaultdict(list)
    for i in range(len(token_list) - n):
        context   = tuple(token_list[i : i + n])  # sliding window of n words
        next_word = token_list[i + n]  # subsequent word after the context word
        model[context].append(next_word)
    return model


ngram_model = build_ngram_model(tokens, n=2) # bigram

print(f"Bigram model built. Unique contexts: {len(ngram_model)}")


def generate_text(seed_text: str, num_words: int, model: dict) -> str:
    # Clean + tokenize seed to match vocabulary in model
    seed_tokens = tokenize(clean_text(seed_text))
    output      = seed_tokens[:] # start with seed words

    # Detect n from model key length
    n = len(next(iter(model)))

    for _ in range(num_words):
        # Build context from the last n words in output
        context = tuple(output[-n:])

        if context in model:
            # Pick a random observed successor for this context
            next_word = random.choice(model[context])

        else:
            # Back-off: search for any context ending with the last word
            last_word  = output[-1]
            candidates = []
            for key, successors in model.items():
                if key[-1] == last_word:
                    candidates.extend(successors)

            if candidates:
                next_word = random.choice(candidates)
            else:
                # fallback: random word from the vocabulary
                next_word = random.choice(tokens)

        output.append(next_word)

    return " ".join(output)



# Demo of the 5 words per str
print("Generate 5 new words per seed:")

test_seeds = [
    "the hare",
    "will you help me"
]

for seed in test_seeds:
    result = generate_text(seed_text=seed, num_words=5, model=ngram_model)
    print(f"Seed : '{seed}'")
    print(f"Result text: {result}")
    print()