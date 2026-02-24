import re
import string

URL = re.compile(r'https?://\S+|www\.\S+')
NON_TEXT = re.compile(r'[^a-zA-Z\s]+')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL.sub("", text)
    text = NON_TEXT.sub(" ", text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_text(text: str) -> str:
    return clean_text(text)
