import re

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    # remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text