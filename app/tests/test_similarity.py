from app.services.analyzer import analyze_texts

def test_similarity_reasonable():
    r = "Experienced Python developer with NLP projects and ML."
    j = "Looking for an NLP engineer with strong Python and machine learning."
    res = analyze_texts(r, j, {"noun_phrases": True, "ner": False})
    assert 0.4 <= res["score"] <= 1.0
    assert "interpretation" in res
