from collections import Counter

def build_vocab(texts):

    counter = Counter()

    for text in texts:

        tokens = text.lower().split()

        counter.update(tokens)

    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    for token in counter:

        vocab[token] = len(vocab)

    return vocab

#test

texts = [
    "i love ai",
    "i study nlp"
]

vocab = build_vocab(texts)

print(vocab)