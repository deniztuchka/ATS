import re
from typing import List, Tuple, Set
from app.nlp.preprocess import doc_from_text

# Basit ama etkili stopword ve yasak ifade listesi
_STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","in","into","is","it",
    "of","on","or","that","the","to","with","will","we","you","our","your","this",
    "their","them","they","he","she","i","was","were","have","has","had","not",
    "but","if","so","than","then","over","under","about","up","down","out"
}
_BAN_PHRASE = {"ideal","candidate","seeking","required","proficient","experience","experiences"}

_TECH_HINT = {
    # tekil teknoloji/skill ipuçları (listeyi genişletebilirsin)
    "python","java","c","c++","matlab","tensorflow","pytorch","docker","kubernetes",
    "aws","gcp","azure","sql","git","linux","pandas","numpy","scikit-learn","flask",
    "django","rest","nlp","ml","ai","bert","spacy","sklearn","spark","hadoop",
    "tableau","powerbi","kafka","airflow","redis","mongodb","postgres","mysql"
}

_token_re = re.compile(r"[a-z0-9]+")

def _normalize_term(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s\+\-#\/\.]", " ", t)  # sadeleştir
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _term_tokens(t: str) -> List[str]:
    return _token_re.findall(t.lower())

def _is_stop_token(tok: str) -> bool:
    return tok in _STOPWORDS or tok in _BAN_PHRASE or len(tok) <= 1

def _is_skill_phrase(term: str) -> bool:
    """
    - Tümü stopword olan veya stopword ile başlayan/bitiyorsa: geçersiz
    - En fazla 4 kelime
    - En az bir teknik/isimvari token içersin
    """
    term = _normalize_term(term)
    toks = _term_tokens(term)
    if not toks:
        return False
    if len(toks) > 4:
        return False
    if _is_stop_token(toks[0]) or _is_stop_token(toks[-1]):
        return False
    # tamamı stopword ise alma
    if all(_is_stop_token(t) for t in toks):
        return False
    # en az bir teknik ipucu veya en az bir 3+ harfli alfa token olsun
    if any(t in _TECH_HINT for t in toks):
        return True
    if any(len(t) >= 3 and t.isalpha() and not _is_stop_token(t) for t in toks):
        return True
    return False

def extract_noun_phrases(text: str) -> List[str]:
    doc = doc_from_text(text)
    out = set()
    for chunk in doc.noun_chunks:
        term = _normalize_term(chunk.text)
        if _is_skill_phrase(term):
            out.add(term)
    return list(out)

def extract_named_entities(text: str) -> List[str]:
    doc = doc_from_text(text)
    whitelist = {"ORG", "PRODUCT", "LANGUAGE"}
    out = set()
    for ent in doc.ents:
        if ent.label_ in whitelist or ent.text.isupper():
            term = _normalize_term(ent.text)
            if _is_skill_phrase(term):
                out.add(term)
    return list(out)

def top_terms_from_vectorizer(vectorizer, tfidf_matrix, doc_index: int, top_k=30) -> List[Tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    row = tfidf_matrix[doc_index].toarray().ravel()
    pairs = [(feature_names[i], float(row[i])) for i in range(len(feature_names)) if row[i] > 0]
    # filtre: skill olmayanı çıkar
    pairs = [(t, w) for (t, w) in pairs if _is_skill_phrase(t)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

def derive_skills(text: str, use_np=True, use_ner=True, seed_terms: List[str] = None) -> Set[str]:
    """
    Skill kümesi döndürür: tekil kelimeler + anlamlı NP/NER.
    """
    skills = set()
    if seed_terms:
        for t in seed_terms:
            t = _normalize_term(t)
            if _is_skill_phrase(t):
                skills.add(t)
    if use_np:
        for t in extract_noun_phrases(text):
            if _is_skill_phrase(t): skills.add(t)
    if use_ner:
        for t in extract_named_entities(text):
            if _is_skill_phrase(t): skills.add(t)
    # ayrıca tekil kelimeler
    for tok in _term_tokens(text):
        if _is_skill_phrase(tok):
            skills.add(tok)
    return skills

def compute_missing_from_sets(job_skills: Set[str], resume_skills: Set[str]):
    missing = sorted(job_skills - resume_skills)
    return [{"term": t, "category": "Skill", "strength": "Absent"} for t in missing]
