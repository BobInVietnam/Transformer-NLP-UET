from data.tokenizer import tokenize

class Vocabulary:

    def __init__(self):

        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<UNK>"
        }

        self.idx = 2

    def build_vocab(self, dataframe):

        for text in dataframe["article"]:

            tokens = tokenize(text)

            for token in tokens:

                if token not in self.word2idx:

                    self.word2idx[token] = self.idx
                    self.idx2word[self.idx] = token

                    self.idx += 1

    def numericalize(self, tokens):

        return [
            self.word2idx.get(token, 1)
            for token in tokens
        ]