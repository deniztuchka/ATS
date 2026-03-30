async function loadPdfResume() {
  const input = document.getElementById("resumePdf");
  const status = document.getElementById("status");
  const resultBadge = document.getElementById("resultBadge");

  if (!input || !input.files || !input.files[0]) {
    alert("Please choose a PDF file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", input.files[0]);

  status.textContent = "Uploading CV PDF and extracting text…";
  status.classList.remove("status-error");

  try {
    const res = await fetch("/api/upload-resume", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      const msg = data && data.error ? data.error : "Error while reading the PDF.";
      status.textContent = "";
      resultBadge.textContent = "PDF error";
      alert(msg);
      return;
    }

    const text = data.resume_text || "";
    document.getElementById("resume").value = text;

    status.textContent = "PDF text loaded into resume area.";
    resultBadge.textContent = "CV loaded from PDF";
  } catch (err) {
    console.error(err);
    status.textContent = "";
    resultBadge.textContent = "PDF error";
    alert("Unexpected error while uploading PDF.");
  }
}

function formatAnalysisResult(data) {
  const scorePercent = data.score ? (data.score * 100).toFixed(2) : "0.00";
  const interpretation = data.interpretation || "Unknown";

  const commonSkills = data.details?.common_skills || [];

  const blockedTerms = [
    "looking",
    "plus",
    "requirements",
    "responsibilities",
    "work",
    "develop",
    "strong",
    "using",
    "systems",
    "science",
    "data",
    "machine",
    "backend",
    "learning responsibilities",
    "requirements strong",
    "responsibilities develop",
    "python - work",
    "python work"
  ];

  const missingSkills = (data.missing_keywords || [])
    .map(item => item.term)
    .filter(term => term && !blockedTerms.includes(term.toLowerCase()))
    .slice(0, 10);

  let summary = "";

  if (interpretation.toLowerCase() === "poor") {
    summary =
      "The resume has a weak match with the job description. Several important skills or keywords are missing.";
  } else if (interpretation.toLowerCase() === "average") {
    summary =
      "The resume has a moderate match with the job description, but some important requirements are still missing.";
  } else if (interpretation.toLowerCase() === "good") {
    summary =
      "The resume matches the job description well and includes many relevant skills.";
  } else {
    summary =
      "The resume was analyzed successfully.";
  }

  const commonSkillsText = commonSkills.length
    ? commonSkills.join(", ")
    : "No strong common skills were found.";

  const missingSkillsText = missingSkills.length
    ? missingSkills.join(", ")
    : "No important missing skills were detected.";

  return `
Resume Analysis Result

Match Score: ${scorePercent}%
Overall Evaluation: ${interpretation}

Matched Skills:
${commonSkillsText}

Missing Important Skills:
${missingSkillsText}

Final Comment:
${summary}
  `.trim();
}

async function analyze() {
  const resume = document.getElementById("resume").value.trim();
  const jobDescription = document.getElementById("jobDescription").value.trim();

  const status = document.getElementById("status");
  const resultBadge = document.getElementById("resultBadge");
  const resultBox = document.getElementById("analysisResult");
  const analyzeBtn = document.getElementById("analyze");

  if (!resume) {
    alert("Please provide resume text (paste or load from PDF).");
    return;
  }

  if (!jobDescription) {
    alert("Please paste the job description.");
    return;
  }

  const payload = {
    resume: resume,
    job_description: jobDescription,
  };

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  status.textContent = "Sending data to backend and waiting for analysis…";
  status.classList.remove("status-error");
  resultBadge.textContent = "Running";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      const msg = data && data.error ? data.error : "Analysis failed.";
      status.textContent = "";
      status.classList.add("status-error");
      status.textContent = msg;
      resultBadge.textContent = "Error";
      resultBox.textContent = msg;
      return;
    }

    resultBox.textContent = formatAnalysisResult(data);

    status.textContent = "Analysis completed successfully.";
    resultBadge.textContent = "Completed";
  } catch (err) {
    console.error(err);
    status.textContent = "";
    status.classList.add("status-error");
    status.textContent =
      "Unexpected error while calling /api/analyze. Check backend logs.";
    resultBadge.textContent = "Error";
    resultBox.textContent =
      "Unexpected error while calling /api/analyze. See browser console and backend logs for details.";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Run Analysis";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyze");
  const loadPdfBtn = document.getElementById("loadPdfBtn");

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", analyze);
  }

  if (loadPdfBtn) {
    loadPdfBtn.addEventListener("click", loadPdfResume);
  }
});