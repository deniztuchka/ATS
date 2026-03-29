from flask import Blueprint, request, jsonify
from app.services.analyzer import analyze_texts
from app.utils.pdf_utils import extract_text_from_pdf

api_bp = Blueprint("api", __name__)

@api_bp.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}

    resume = (data.get("resume") or "").strip()
    job_description = (data.get("job_description") or "").strip()

    if not resume or not job_description:
        return jsonify({"error": "Resume and Job Description are required."}), 400

    options = {
        "language": "en",
        "model": "default",
        "include_missing_skills": True,
        "include_score": True,
        "noun_phrases": True,
        "ner": True,
    }

    try:
        result = analyze_texts(resume, job_description, options)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Analysis crashed: {str(e)}"}), 500


@api_bp.post("/upload-resume")
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    try:
        text = extract_text_from_pdf(file)
        return jsonify({"resume_text": text}), 200
    except Exception as e:
        return jsonify({"error": f"PDF reading failed: {str(e)}"}), 500