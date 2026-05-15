"""
keywords.py — general-purpose skill extraction using spaCy POS tags.

Works for any industry: IT, fashion, medicine, cooking, marketing, etc.

Key improvements over original:
  1. Removed raw token loop — was adding every 4+ letter word as a "skill"
  2. POS tag filtering — only keeps noun chunks whose HEAD is NOUN or PROPN
  3. Special term recovery — CI/CD, C++, DevOps etc. preserved via preprocess
  4. No hardcoded tech whitelist — domain-neutral
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
    # HR boilerplate
    "passionate","driven","motivated","excited","dynamic","innovative","creative",
    "talented","ambitious","enthusiastic","proactive","versatile","detail",
    "oriented","self","starter","player","fast","learner","quick",
    "growing","startup","culture","mission","vision","values",
    "opportunity","career","opening","join","hire","hiring",
    "apply","send","cover","letter","interview","candidate","ideal",
    # Benefits / compensation
    "salary","compensation","pay","bonus","equity","stock","option","benefits",
    "benefit","vacation","pto","holiday","insurance","dental","flexible",
    "remote","hybrid","location","city","country","travel",
    # Generic descriptors
    "good","great","best","strong","solid","excellent","proficient","able",
    "ability","capable","understanding","familiarity","awareness",
    "background","exposure","basis","level","high","low","minimum","maximum",
    "plus","required","preferred","nice","must","should","would","could",
    "also","well","very","highly","etc","min","max",
    # Generic nouns that are never skills
    "responsibility","responsibilities","requirement","requirements",
    "skill","skills","tool","tools","technology","technologies",
    "solution","solutions","system","systems","application","applications",
    "platform","platforms","process","processes","project","projects",
    "product","products","service","services","environment","environments",
    "team","teams","member","members","colleague","stakeholder",
    "client","customer","user","users","business","company","organization",
    "department","management","lead","leader","leadership",
    "communication","collaboration","coordination","presentation","reporting",
    "experience","experienced","knowledge","role","position","company",
    # Action verbs
    "work","works","working","making","using","need","needs","want","take",
    "give","help","support","maintain","manage","ensure","drive","build",
    "define","review","identify","create","design","test","write","run",
    "develop","implement","integrate","automate","perform","conduct",
    "collaborate","reside","require","fluent","willingness","residing",
    # Location / language requirements (not skills)
    "english","polish","french","german","spanish","dutch","italian",
    "poland","germany","france","netherlands","sweden","norway",
    "international","fluent","native","speaker",
    # Numbers / time
    "year","years","month","months","day","days","hour","hours",
}

# POS tags for genuine skill head words
_SKILL_POS = {"NOUN", "PROPN"}

# POS tags that indicate noise
_NOISE_POS = {"ADJ", "ADV", "VERB", "AUX", "DET", "PART", "PUNCT"}

# Canonical display names for normalised special terms
_DISPLAY_NAMES = {
    "cicd":       "CI/CD",
    "cpp":        "C++",
    "csharp":     "C#",
    "dotnet":     ".NET",
    "aspnet":     "ASP.NET",
    "nodejs":     "Node.js",
    "nextjs":     "Next.js",
    "vuejs":      "Vue.js",
    "reactjs":    "React.js",
    "scikitlearn":"scikit-learn",
}

_token_re = re.compile(r"[a-z0-9][a-z0-9\+\-#\.\/]*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_term(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\s\+\-#\/\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _display_term(t: str) -> str:
    """Return the human-readable form of a normalised term."""
    return _DISPLAY_NAMES.get(t, t)


def _term_tokens(t: str) -> List[str]:
    return _token_re.findall(t.lower())


def _is_stopword(tok: str) -> bool:
    return tok.lower() in _STOPWORDS or len(tok) <= 1


def _is_valid_term(term: str) -> bool:
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
    Return True if a spaCy noun chunk represents a real skill or tool.

    Rules:
      1. HEAD token must be NOUN or PROPN
         (drops "passionate engineer", "growing team", "strong communication")
      2. Must pass structural stopword check
      3. Must not be dominated by noise POS tokens
    """
    if chunk.root.pos_ not in _SKILL_POS:
        return False
    if not _is_valid_term(chunk.text):
        return False
    tokens = list(chunk)
    noise = sum(1 for t in tokens if t.pos_ in _NOISE_POS and not t.is_stop)
    if noise > len(tokens) / 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases filtered by POS head tag.
    Works for any industry domain.
    """
    doc = doc_from_text(text)
    out: Set[str] = set()
    for chunk in doc.noun_chunks:
        if _chunk_is_skill(chunk):
            term = _normalize_term(chunk.text)
            if _is_valid_term(term):
                out.add(_display_term(term))
    return list(out)


def extract_named_entities(text: str) -> List[str]:
    """
    Extract named entities: ORG, PRODUCT, LANGUAGE.
    Catches tools, frameworks, software names.
    """
    doc = doc_from_text(text)
    whitelist_labels = {"ORG", "PRODUCT", "LANGUAGE"}
    out: Set[str] = set()
    for ent in doc.ents:
        if ent.label_ in whitelist_labels or ent.text.isupper():
            term = _normalize_term(ent.text)
            if _is_valid_term(term):
                out.add(_display_term(term))
    return list(out)


def top_terms_from_vectorizer(
    vectorizer, tfidf_matrix, doc_index: int, top_k: int = 30
) -> List[Tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    row = tfidf_matrix[doc_index].toarray().ravel()
    pairs = [
        (_display_term(feature_names[i]), float(row[i]))
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
      1. NER  — named entities (ORG, PRODUCT, LANGUAGE)
      2. Noun phrases filtered by POS head
      3. TF-IDF seeds (structurally validated)

    The raw token loop from the original has been removed — it was
    adding every 4+ letter word regardless of grammatical role.
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
                skills.add(_display_term(norm))

    return skills


def compute_missing_from_sets(
    job_skills: Set[str], resume_skills: Set[str]
) -> List[dict]:
    missing = sorted(job_skills - resume_skills)
    return [{"term": t, "category": "Skill", "strength": "Absent"} for t in missing]
