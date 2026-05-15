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

# Special terms to normalise before stripping punctuation.
# Maps the raw form to the canonical lowercase form we want to keep.
_SPECIAL_TERMS = {
    r"ci/cd":          "cicd",
    r"c\+\+":          "cpp",
    r"c#":             "csharp",
    r"\.net":          "dotnet",
    r"asp\.net":       "aspnet",
    r"node\.js":       "nodejs",
    r"next\.js":       "nextjs",
    r"vue\.js":        "vuejs",
    r"react\.js":      "reactjs",
    r"scikit-learn":   "scikitlearn",
    r"devops":         "devops",
    r"mlops":          "mlops",
}

def _protect_special_terms(text: str) -> str:
    """Replace special-character terms with safe placeholders before normalisation."""
    for pattern, replacement in _SPECIAL_TERMS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_for_vectorizer(text: str) -> str:
    """Lowercase + keep only letters/digits/basic punctuation; condense spaces."""
    text = clean_whitespace(text)
    text = _protect_special_terms(text)          # protect CI/CD, C++, etc.
    text = re.sub(r"[^A-Za-z0-9\s\.,:/+\-#()]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@lru_cache(maxsize=128)
def doc_from_text(text: str):
    """Run spaCy on the original (un-normalised) text for NER and noun chunks."""
    nlp = get_nlp()
    return nlp(text)
