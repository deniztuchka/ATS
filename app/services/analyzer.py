from app.nlp.preprocess import normalize_for_vectorizer   # <-- ÖNEMLİ: bu import gerekiyordu
from app.nlp.vectorizer import build_vectorizer
from app.nlp.keywords import (
    top_terms_from_vectorizer,
    derive_skills,
    compute_missing_from_sets
)

def interpret(score: float) -> str:
    if score < 0.40: return "Poor"
    if score < 0.60: return "Fair"
    if score < 0.80: return "Good"
    return "Excellent"

def jaccard(a_set, b_set) -> float:
    if not a_set and not b_set:
        return 0.0
    union = a_set | b_set
    inter = a_set & b_set
    return len(inter) / max(1, len(union))

def analyze_texts(resume: str, job: str, options: dict):
    # 1) Normalize
    resume_norm = normalize_for_vectorizer(resume)
    job_norm = normalize_for_vectorizer(job)

    # 2) TF-IDF (seed terimler için; hata olursa atla)
    vectorizer = build_vectorizer()
    top_job, top_resume = [], []
    try:
        tfidf = vectorizer.fit_transform([resume_norm, job_norm])
        top_job = top_terms_from_vectorizer(vectorizer, tfidf, 1, top_k=30)
        top_resume = top_terms_from_vectorizer(vectorizer, tfidf, 0, top_k=30)
    except Exception:
        top_job, top_resume = [], []

    # 3) Skill kümeleri (NP/NER + seed)
    seed_job = [t for t, _ in top_job]
    seed_resume = [t for t, _ in top_resume]

    job_skills = derive_skills(
        job,
        use_np=options.get("noun_phrases", True),
        use_ner=options.get("ner", True),
        seed_terms=seed_job
    )
    resume_skills = derive_skills(
        resume,
        use_np=options.get("noun_phrases", True),
        use_ner=options.get("ner", True),
        seed_terms=seed_resume
    )

    # 4) Jaccard skill overlap skoru + sadece skill-based missing
    score = jaccard(resume_skills, job_skills)
    missing = compute_missing_from_sets(job_skills, resume_skills)

    return {
        "score": round(float(score), 4),
        "interpretation": interpret(score),
        "missing_keywords": missing,
        "details": {
            "method": "skill_overlap_jaccard",
            "overlap_count": len(resume_skills & job_skills),
            "resume_skill_count": len(resume_skills),
            "job_skill_count": len(job_skills),
            "common_skills": sorted(list(resume_skills & job_skills))[:20],
            "top_job_terms": top_job[:12],
            "top_resume_terms": top_resume[:12]
        }
    }
