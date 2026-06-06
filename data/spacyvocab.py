from collections import Counter
import spacy

class Vocabulary:
    def __init__(self, max_vocab=30000):
        self.max_vocab = max_vocab
        # Initialize standard special tokens matching your model constraints
        self.itos = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        
        # Initialize a blank Vietnamese language object
        # This acts purely as a fast rule-based tokenizer
        self.nlp = spacy.blank("vi")

    def tokenize_text(self, text):
        """
        Tokenizes Vietnamese text. Multi-syllable terms are bonded with underscores
        (e.g., "máy tính" -> "máy_tính") and punctuation is isolated safely.
        """
        doc = self.nlp(text.lower().strip())
        return [token.text for token in doc]

    def build_vocab(self, sentences):
        """Compiles unique token indices sorted by corpus frequency."""
        frequencies = Counter()
        for sentence in sentences:
            tokens = self.tokenize_text(sentence)
            frequencies.update(tokens)
            
        sorted_words = sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Enforce your exact vocabulary cap constraints
        sorted_words = sorted_words[: self.max_vocab - 4]
        
        idx = 4
        for word, freq in sorted_words:
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def numericalize(self, tokens):
        """Maps a list of string tokens into an array of absolute integer IDs."""
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]