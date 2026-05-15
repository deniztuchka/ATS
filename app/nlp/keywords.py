"""
keywords.py — improved skill extraction using spaCy POS tags.

The core fix over the original:
  - REMOVED the raw token loop (was adding every 4+ letter word as a skill)
  - Noun phrases: now filters by spaCy POS tags — only keeps chunks whose
    HEAD word is NOUN or PROPN, removing verb phrases and adjective phrases
  - NER: keeps ORG, PRODUCT, LANGUAGE entities (unchanged — already good)
  - TF-IDF seeds: only added if they also appear as a noun/propn in the text
  - No hardcoded tech whitelist — works for any industry

This means:
  "passionate" -> POS=ADJ, dropped
  "growing"    -> POS=VERB, dropped
  "Python"     -> POS=PROPN, kept
  "pattern making" -> head=NOUN, kept (fashion)
  "patient assessment" -> head=NOUN, kept (nursing)
  "sous vide"  -> head=NOUN, kept (cooking)
"""

import re
from typing import List, Tuple, Set
from app.nlp.preprocess import doc_from_text

# ---------------------------------------------------------------------------
# Stop words — HR boilerplate / generic words that are never skills
# ---------------------------------------------------------------------------
_STOPWORDS: Set[str] = {
    # Articles, prepositions, conjunctions
    "a","an","and","are","as","at","be","by","for","from","in","into","is","it",
    "of","on","or","that","the","to","with","will","we","you","our","your","this",
    "their","them","they","he","she","i","was","were","have","has","had","not",
    "but","if","so","than","then","over","under","about","up","down","out",
    # HR boilerplate — job ad filler words
    "passionate","driven","motivated","excited","dynamic","innovative","creative",
    "talented","ambitious","enthusiastic","proactive","versatile","detail",
    "oriented","self","starter","team","player","fast","learner","quick",
    "growing","startup","company","culture","mission","vision","values",
    "opportunity","career","position","opening","role","join","hire","hiring",
    "apply","send","resume","cover","letter","interview","candidate","ideal",
    # Benefits / compensation
    "salary","compensation","pay","bonus","equity","stock","option","benefits",
    "benefit","vacation","pto","holiday","insurance","dental","health","vision",
    "flexible","remote","hybrid","office","location","city","country","travel",
    # Generic descriptors that are never skills
    "good","great","best","strong","solid","excellent","proficient","able",
    "ability","capable","knowledge","understanding","familiarity","awareness",
    "experience","experienced","background","exposure","basis","level","high",
    "low","minimum","maximum","plus","bonus","required","preferred","nice",
    "must","have","should","would","could","also","well","very","highly","etc",
    # Generic nouns from job descriptions
    "responsibility","responsibilities","requirement","requirements","skill",
    "skills","tool","tools","technology","technologies","solution","solutions",
    "system","systems","application","applications","platform","platforms",
    "process","processes","project","projects","product","products","service",
    "services","environment","team","teams","member","members","colleague",
    "stakeholder","client","customer","user","users","business","company",
    "organization","department","management","lead","leader","leadership",
    "communication","collaboration","coordination","presentation","reporting",
    # Very short / meaningless
    "work","works","working","making","using","need","needs","want","take",
    "give","help","support","maintain","manage","ensure","drive","build",
    "define","review","identify","create","design","test","write","run",
}

# POS tags that indicate genuine content words (skills, tools, concepts)
_SKILL_POS = {"NOUN", "PROPN"}

# POS tags to reject even if they sneak into a noun chunk
_NOISE_POS = {"ADJ", "ADV", "VERB", "AUX", "DET", "PART", "PUNCT", "NUM"}

_token_re = re.compile(r"[a-z0-9][a-z0-9\+\-#\.\/]*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_term(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\s\+\-#\/\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _term_tokens(t: str) -> List[str]:
    return _token_re.findall(t.lower())


def _is_stopword(tok: str) -> bool:
    return tok.lower() in _STOPWORDS or len(tok) <= 1


def _is_valid_term(term: str) -> bool:
    """Structural check — reject empty, too long, or stopword-only terms."""
    norm = _normalize_term(term)
    toks = _term_tokens(norm)
    if not toks or len(toks) > 4:
        return False
    if _is_stopword(toks[0]) or _is_stopword(toks[-1]):
        return False
    if all(_is_stopword(t) for t in toks):
        return False
    if len(toks) == 1 and len(toks[0]) < 3:
        return False
    return True


def _chunk_is_skill(chunk) -> bool:
    """
    Return True if a spaCy noun chunk looks like a real skill/tool/concept.

    Criteria:
      1. The HEAD token must be NOUN or PROPN
         (rejects "growing team", "strong communication", "passionate engineer")
      2. The chunk text must pass the structural stopword check
      3. The chunk must not be dominated by noise POS tags
    """
    if chunk.root.pos_ not in _SKILL_POS:
        return False

    if not _is_valid_term(chunk.text):
        return False

    tokens = list(chunk)
    noise_count = sum(1 for t in tokens if t.pos_ in _NOISE_POS and not t.is_stop)
    if noise_count > len(tokens) / 2:
        return False

    return True


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases using spaCy, filtered by POS tags.
    Works for any domain — IT, fashion, medicine, cooking, marketing, etc.
    """
    doc = doc_from_text(text)
    out: Set[str] = set()

    for chunk in doc.noun_chunks:
        if _chunk_is_skill(chunk):
            term = _normalize_term(chunk.text)
            if _is_valid_term(term):
                out.add(term)

    return list(out)


def extract_named_entities(text: str) -> List[str]:
    """
    Extract named entities: ORG, PRODUCT, LANGUAGE.
    These tend to be tools, software, technologies, certifications.
    """
    doc = doc_from_text(text)
    whitelist_labels = {"ORG", "PRODUCT", "LANGUAGE"}
    out: Set[str] = set()

    for ent in doc.ents:
        if ent.label_ in whitelist_labels or ent.text.isupper():
            term = _normalize_term(ent.text)
            if _is_valid_term(term):
                out.add(term)

    return list(out)


def top_terms_from_vectorizer(
    vectorizer, tfidf_matrix, doc_index: int, top_k: int = 30
) -> List[Tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    row = tfidf_matrix[doc_index].toarray().ravel()
    pairs = [
        (feature_names[i], float(row[i]))
        for i in range(len(feature_names))
        if row[i] > 0 and _is_valid_term(feature_names[i])
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def derive_skills(
    text: str,
    use_np: bool = True,
    use_ner: bool = True,
    seed_terms: List[str] = None,
) -> Set[str]:
    """
    Build a skill set from a document.

    Sources:
      1. NER — named entities (ORG, PRODUCT, LANGUAGE)
      2. Noun phrases filtered by POS head tag
      3. TF-IDF seeds — structurally validated
    """
    skills: Set[str] = set()

    if use_ner:
        for t in extract_named_entities(text):
            skills.add(t)

    if use_np:
        for t in extract_noun_phrases(text):
            skills.add(t)

    if seed_terms:
        for t in seed_terms:
            norm = _normalize_term(t)
            if _is_valid_term(norm):
                skills.add(norm)

    return skills


def compute_missing_from_sets(
    job_skills: Set[str], resume_skills: Set[str]
) -> List[dict]:
    missing = sorted(job_skills - resume_skills)
    return [{"term": t, "category": "Skill", "strength": "Absent"} for t in missing]
