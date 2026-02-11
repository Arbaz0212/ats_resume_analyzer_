"""ATS Resume Analyzer — FastAPI application with LLM integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    MAX_RESUMES, MAX_FILE_SIZE_MB, MAX_JD_CHARS, MIN_JD_CHARS,
    FEATURE_JD_FILE_UPLOAD, FEATURE_OPTIONAL_ROLE,
)
from app.services.pdf_parser import extract_text
from app.services.scoring_engine import analyze_batch
from app.services.skill_matcher import get_available_roles
from app.services.llm_service import ai_enabled, parse_jd

import re


def _sanitize_filename(name: str) -> str:
    """Strip dangerous characters from uploaded filenames."""
    return re.sub(r'[^\w\-. ]', '_', name or "unnamed")[:255]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ATS Resume Analyzer", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8899",
        "http://127.0.0.1:8899",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error_html(request: Request, message: str) -> HTMLResponse:
    """Return an error as an HTML partial that HTMX can swap in."""
    return templates.TemplateResponse(
        "partials/error_banner.html",
        {"request": request, "error_message": message},
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "roles": get_available_roles(),
        "ai_enabled": ai_enabled(),
    })


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "roles": get_available_roles(),
        "ai_enabled": ai_enabled(),
    })


# ---------------------------------------------------------------------------
# API routes (return HTML partials for HTMX)
# ---------------------------------------------------------------------------
@app.post("/api/analyze", response_class=HTMLResponse)
async def api_analyze(
    request: Request,
    resumes: Annotated[list[UploadFile], File(...)],
    job_description: Annotated[str | None, Form()] = None,
    job_description_file: Annotated[UploadFile | None, File()] = None,
    role: Annotated[str | None, Form()] = None,
):
    """Analyze uploaded resumes against JD. Returns HTML partial for HTMX.

    JD Input (one required):
    - job_description: Paste JD text directly
    - job_description_file: Upload JD as PDF/DOCX (if FEATURE_JD_FILE_UPLOAD enabled)

    Role (optional if FEATURE_OPTIONAL_ROLE enabled):
    - If provided, used as metadata/advisory only
    - Scoring uses unified weights regardless of role selection
    """
    # --- Extract JD text from file or form field ---
    jd_text: str = ""

    if job_description_file and job_description_file.filename and FEATURE_JD_FILE_UPLOAD:
        # JD uploaded as file
        if job_description and job_description.strip():
            return _error_html(request, "Provide JD via paste OR upload, not both.")
        try:
            jd_text = await extract_text(job_description_file)
        except ValueError as exc:
            return _error_html(request, f"JD file error: {exc}")
        except Exception:
            logger.exception("Error parsing JD file %s", job_description_file.filename)
            return _error_html(request, "Failed to parse JD file. Use PDF or DOCX.")
    elif job_description:
        # JD pasted as text
        jd_text = job_description.strip()
    else:
        return _error_html(request, "Job description is required. Paste text or upload PDF/DOCX.")

    # --- Validate JD length ---
    if len(jd_text) < MIN_JD_CHARS:
        return _error_html(request, f"Job description too short (min {MIN_JD_CHARS} characters).")
    if len(jd_text) > MAX_JD_CHARS:
        jd_text = jd_text[:MAX_JD_CHARS]
        logger.warning("JD truncated to %d chars", MAX_JD_CHARS)

    # --- Validate role (optional if feature flag enabled) ---
    if not FEATURE_OPTIONAL_ROLE:
        if not role or not role.strip():
            return _error_html(request, "Please select a role.")

    # Normalize role - empty string or None both mean "no role selected"
    role_normalized = (role or "").strip() or None

    # --- Validate resumes ---
    if not resumes or (len(resumes) == 1 and not resumes[0].filename):
        return _error_html(request, "Please upload at least one resume.")
    if len(resumes) > MAX_RESUMES:
        return _error_html(request, f"Maximum {MAX_RESUMES} resumes allowed.")

    # --- Extract text from each resume ---
    parsed_resumes: list[dict[str, str]] = []
    errors: list[str] = []

    for resume_file in resumes:
        try:
            resume_file.file.seek(0, 2)
            size_mb = resume_file.file.tell() / (1024 * 1024)
            resume_file.file.seek(0)
            if size_mb > MAX_FILE_SIZE_MB:
                errors.append(f"{resume_file.filename}: exceeds {MAX_FILE_SIZE_MB}MB limit")
                continue

            text = await extract_text(resume_file)
            if not text.strip():
                errors.append(f"{resume_file.filename}: could not extract text")
                continue

            parsed_resumes.append({
                "filename": _sanitize_filename(resume_file.filename),
                "text": text,
            })
        except ValueError as exc:
            errors.append(f"{resume_file.filename}: {exc}")
        except Exception:
            logger.exception("Error parsing %s", resume_file.filename)
            errors.append(f"{resume_file.filename}: parsing error")

    if not parsed_resumes:
        error_detail = "No valid resumes could be parsed."
        if errors:
            error_detail += " Issues: " + "; ".join(errors)
        return _error_html(request, error_detail)

    # --- Run analysis (async with AI) ---
    results = await analyze_batch(parsed_resumes, jd_text, role_normalized)
    results["errors"] = errors

    return templates.TemplateResponse("partials/results_content.html", {
        "request": request,
        "results": results,
        "jd_text": jd_text,
    })


@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request):
    return templates.TemplateResponse("review.html", {
        "request": request,
        "ai_enabled": ai_enabled(),
    })


@app.post("/api/review", response_class=HTMLResponse)
async def api_review(
    request: Request,
    resume: Annotated[UploadFile, File(...)],
    job_description: Annotated[str | None, Form()] = None,
    job_description_file: Annotated[UploadFile | None, File()] = None,
):
    """Individual resume review — single resume, optional JD."""
    from app.services.scoring_engine import analyze_single_resume

    # Extract JD (optional)
    jd_text: str | None = None
    if job_description_file and job_description_file.filename:
        try:
            jd_text = await extract_text(job_description_file)
        except Exception:
            pass
    elif job_description and job_description.strip():
        jd_text = job_description.strip()

    # Validate resume
    if not resume or not resume.filename:
        return _error_html(request, "Please upload a resume.")

    resume.file.seek(0, 2)
    size_mb = resume.file.tell() / (1024 * 1024)
    resume.file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return _error_html(request, f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    try:
        resume_text = await extract_text(resume)
    except Exception:
        return _error_html(request, "Could not parse resume. Use PDF or DOCX.")

    if not resume_text.strip():
        return _error_html(request, "Could not extract text from resume.")

    # Run analysis
    result = await analyze_single_resume(
        resume_text, jd_text, filename=resume.filename or "resume",
    )

    has_jd = bool(jd_text and jd_text.strip())
    score = result.get("final_score", 0)

    return templates.TemplateResponse("partials/review_results.html", {
        "request": request,
        "result": result,
        "score": score,
        "has_jd": has_jd,
        "ai_enabled": ai_enabled(),
    })


@app.post("/api/parse-jd", response_class=HTMLResponse)
async def api_parse_jd(
    request: Request,
    job_description: Annotated[str | None, Form()] = None,
    job_description_file: Annotated[UploadFile | None, File()] = None,
):
    """Parse JD and return a preview of extracted requirements."""
    # Extract JD text (same logic as analyze)
    jd_text = ""
    if job_description_file and job_description_file.filename:
        try:
            jd_text = await extract_text(job_description_file)
        except Exception:
            jd_text = ""
    elif job_description:
        jd_text = job_description.strip()

    if len(jd_text) < MIN_JD_CHARS:
        return _error_html(request, f"JD too short for preview (min {MIN_JD_CHARS} chars).")

    parsed = None
    if ai_enabled():
        parsed = await parse_jd(jd_text[:MAX_JD_CHARS])

    return templates.TemplateResponse("partials/jd_parsed_preview.html", {
        "request": request,
        "parsed": parsed,
    })


@app.get("/api/roles")
async def api_roles():
    return {"roles": get_available_roles()}


@app.get("/api/jd-templates")
async def api_jd_templates():
    """Return pre-built JD templates for each role."""
    import json
    tpl_path = BASE_DIR / "data" / "jd_templates.json"
    with open(tpl_path, "r", encoding="utf-8") as f:
        return json.load(f)
