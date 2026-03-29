from app.nlp.preprocess import normalize_for_vectorizer

def test_normalize_basic():
    s = "Experienced in Developing Python applications!! "
    out = normalize_for_vectorizer(s)
    assert "python" in out
    assert out == out.lower()
