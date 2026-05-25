from collections import Counter


class Vocabulary:

    def __init__(self, max_vocab=30000):

        self.max_vocab = max_vocab

        self.itos = {

            0: "<PAD>",
            1: "<UNK>",
            2: "<SOS>",
            3: "<EOS>"
        }

        self.stoi = {

            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3
        }

    def build_vocab(self, sentences):

        frequencies = Counter()

        for sentence in sentences:

            sentence = sentence.lower()

            tokens = sentence.split()

            frequencies.update(tokens)

        sorted_words = sorted(

            frequencies.items(),

            key=lambda x: x[1],

            reverse=True
        )

        sorted_words = sorted_words[
            : self.max_vocab  - 4
        ]

        idx = 4

        for word, freq in sorted_words:

            self.stoi[word] = idx

            self.itos[idx] = word

            idx += 1

    def numericalize(self, tokens):

        return [

            self.stoi.get(
                token.lower(),
                self.stoi["<UNK>"]
            )

            for token in tokens
        ]

