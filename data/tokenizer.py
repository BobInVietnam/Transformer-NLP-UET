def tokenize(text):

    text = text.lower()

    return text.split()

def tokens_to_ids(tokens, vocab):

    ids = []

    for token in tokens:

        ids.append(
            vocab.get(token, vocab["<unk>"])
        )

    return ids

# Vocabulary
vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "i": 2,
    "love": 3,
    "ai": 4
}

# Test
tokens = ["i", "love", "ai"]

ids = tokens_to_ids(tokens, vocab)

print(ids)