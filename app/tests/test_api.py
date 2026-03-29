from app.main import create_app

def test_api_analyze_client():
    app = create_app()
    client = app.test_client()
    payload = {
        "resume": "Python developer with pandas and numpy.",
        "job": "We need a data engineer with Python and SQL skills.",
        "options": {"noun_phrases": True, "ner": True}
    }
    rv = client.post("/api/analyze", json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "score" in data and "missing_keywords" in data
