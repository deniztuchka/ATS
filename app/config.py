import os

class AppConfig:
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "50000"))
    ENABLE_NER = os.getenv("ENABLE_NER", "true").lower() == "true"
