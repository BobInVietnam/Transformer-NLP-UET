from collections import Counter
from pyvi import ViTokenizer  # Import the Vietnamese word-segmentation library

class PyViVocabulary:
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
            
            # 1. INTEGRATION: Pre-segment the sentence using pyvi.
            # This turns "học sinh học máy tính" into "học_sinh học máy_tính"
            segmented_sentence = ViTokenizer.tokenize(sentence)
            
            # 2. Split by whitespace now that compound phrases are safely joined
            tokens = segmented_sentence.split()
            
            frequencies.update(tokens)
            
        # Sort words based on frequency counts
        sorted_words = sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Enforce your vocabulary capacity constraints
        sorted_words = sorted_words[: self.max_vocab - 4]
        
        idx = 4
        for word, freq in sorted_words:
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def numericalize(self, tokens):
        """
        Expects a pre-split list of tokens (e.g. from segmented_sentence.split()).
        """
        return [
            self.stoi.get(
                token.lower(),
                self.stoi["<UNK>"]
            )
            for token in tokens
        ]
        
if __name__ == "__main__":
    # 1. Create a dummy corpus of raw Vietnamese sentences
    corpus = [
        "Học sinh học khoa học máy tính tại trường đại học.",
        "Máy tính là một công cụ công nghệ mạnh mẽ.",
        "Khoa học công nghệ đang phát triển đột phá."
    ]
    
    # 2. Instantiate and build the vocabulary using the upgraded pyvi logic
    v = PyViVocabulary(max_vocab=100)
    v.build_vocab(corpus)
    
    # 3. Check what words made it into the token collection registry
    print("--- Extracted String-to-Index Map (stoi) ---")
    for word, index in list(v.stoi.items())[:15]:
        print(f"'{word}': {index}")
        
    print("\n--- Testing Numericalization Pipeline ---")
    test_sentence = "Học sinh sử dụng máy tính."
    
    # Remember: You must tokenize with pyvi whenever processing text for this model!
    segmented_test = ViTokenizer.tokenize(test_sentence)
    test_tokens = segmented_test.split()
    
    print("Raw text:       ", test_sentence)
    # Output: học_sinh sử_dụng máy_tính .
    print("Tokens split:   ", test_tokens) 
    
    # Convert token arrays to absolute structural ID vectors
    numerical_vector = v.numericalize(test_tokens)
    print("Numerical IDs:  ", numerical_vector)