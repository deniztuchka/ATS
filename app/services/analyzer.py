from app.nlp.preprocess import normalize_for_vectorizer
from app.nlp.vectorizer import build_vectorizer
from app.nlp.keywords import (
    top_terms_from_vectorizer,
    derive_skills,
    compute_missing_from_sets,
)

def interpret(score: float) -> str:
    if score < 0.40: return "Poor"
    if score < 0.60: return "Fair"
    if score < 0.80: return "Good"
    return "Excellent"

def coverage_score(resume_skills, job_skills) -> float:
    """
    Recall-based: what fraction of the job's skills appear in the resume.
    Does NOT penalise the candidate for having more skills than the job asks for.
    Real ATS systems score this way.
    """
    if not job_skills:
        return 0.0
    matched = resume_skills & job_skills
    return len(matched) / len(job_skills)

def analyze_texts(resume: str, job: str, options: dict):
    resume_norm = normalize_for_vectorizer(resume)
    job_norm = normalize_for_vectorizer(job)

    vectorizer = build_vectorizer()
    top_job, top_resume = [], []
    try:
        tfidf = vectorizer.fit_transform([resume_norm, job_norm])
        top_job = top_terms_from_vectorizer(vectorizer, tfidf, 1, top_k=30)
        top_resume = top_terms_from_vectorizer(vectorizer, tfidf, 0, top_k=30)
    except Exception:
        top_job, top_resume = [], []

    # Deduplicate seed terms (unigrams only after vectorizer fix)
    seed_job = list(dict.fromkeys(t for t, _ in top_job))
    seed_resume = list(dict.fromkeys(t for t, _ in top_resume))

    job_skills = derive_skills(
        job,
        use_np=options.get("noun_phrases", True),
        use_ner=options.get("ner", True),
        seed_terms=seed_job,
    )
    resume_skills = derive_skills(
        resume,
        use_np=options.get("noun_phrases", True),
        use_ner=options.get("ner", True),
        seed_terms=seed_resume,
    )

    # Both are sets — exact duplicates already removed automatically
    score = coverage_score(resume_skills, job_skills)
    matched = sorted(resume_skills & job_skills)
    missing = compute_missing_from_sets(job_skills, resume_skills)

    return {
        "score": round(float(score), 4),
        "interpretation": interpret(score),
        "comment": "",
        "matched_skills": matched,
        "missing_keywords": missing,
        "details": {
            "method": "coverage_recall_score",
            "overlap_count": len(matched),
            "resume_skill_count": len(resume_skills),
            "job_skill_count": len(job_skills),
            "common_skills": matched,
            "top_job_terms": top_job[:12],
            "top_resume_terms": top_resume[:12],
        }
    }
