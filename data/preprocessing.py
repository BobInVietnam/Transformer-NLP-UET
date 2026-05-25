import re

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

def tokenize(text):

    return text.split()

def text_pipeline(text, vocab):

    # preprocess
    text = preprocess_text(text)
    # tokenize
    tokens = tokenize(text)
    # token -> id
    ids = vocab.numericalize(tokens)

    return ids