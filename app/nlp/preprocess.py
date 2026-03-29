import re
import spacy
from functools import lru_cache
from app.config import AppConfig

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(AppConfig.SPACY_MODEL, disable=[])
    return _nlp

def clean_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def normalize_for_vectorizer(text: str) -> str:
    """Lowercase + keep only letters/digits/basic punctuation; condense spaces."""
    text = clean_whitespace(text)
    # keep letters, digits, basic punctuation
    text = re.sub(r"[^A-Za-z0-9\s\.,:/+\-#()]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@lru_cache(maxsize=128)
def doc_from_text(text: str):
    nlp = get_nlp()
    return nlp(text)
