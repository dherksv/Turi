import re
import unicodedata
from datetime import datetime


def normalize(text: str) -> dict:
    """
    Clean raw user input and return a normalized payload
    with metadata the rest of the pipeline uses.
    """

    # 1. unicode normalize — handles weird quotes, em-dashes, accents
    text = unicodedata.normalize("NFKC", text)

    # 2. strip control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. collapse excess whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return {
        "text":       text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "timestamp":  datetime.utcnow().isoformat(),
    }