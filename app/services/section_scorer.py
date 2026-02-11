"""JD-aware section scoring for experience, education, and projects.

Every score is computed RELATIVE to what the Job Description asks for.
A PhD is only valuable if the JD values it. 10 years experience is only
valuable if the JD asks for senior-level candidates.
"""

from __future__ import annotations

import re

# Seniority level ordering (higher = more senior)
_SENIORITY_MAP = {"unknown": 0, "intern": 1, "junior": 2, "mid": 3, "senior": 4}


# ---------------------------------------------------------------------------
# JD Requirement Extraction
# ---------------------------------------------------------------------------

def _extract_required_years(jd_text: str) -> int | None:
    """Extract minimum years of experience required from the JD."""
    jd_lower = jd_text.lower()
    patterns = [
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s*(?:of)?\s*(?:experience|exp)",
        r"(?:at least|minimum|min)\s*(\d{1,2})\+?\s*(?:years?|yrs?)",
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s*(?:in|of|working|relevant)",
    ]
    years = []
    for p in patterns:
        years.extend(int(m) for m in re.findall(p, jd_lower))
    return max(years) if years else None


def _extract_seniority_level(jd_text: str) -> str:
    """Detect the seniority level the JD is hiring for."""
    jd_lower = jd_text.lower()
    if re.search(r"\b(?:senior|sr\.?|lead|staff|principal|architect)\b", jd_lower):
        return "senior"
    if re.search(r"\b(?:junior|jr\.?|entry.?level|graduate|new grad|fresher)\b", jd_lower):
        return "junior"
    if re.search(r"\b(?:intern|internship|trainee|apprentice|co.?op)\b", jd_lower):
        return "intern"
    return "mid"  # default assumption


def _extract_required_education(jd_text: str) -> str:
    """Detect the education level the JD requires.

    Order matters: check PhD first, then masters, then bachelors.
    Each pattern must be specific enough to avoid false positives like
    "Scrum Master" or "master data management" triggering masters.
    """
    jd_lower = jd_text.lower()
    if re.search(r"\b(?:ph\.?d|doctorate|doctoral)\b", jd_lower):
        return "phd"
    # Masters: require degree-like context (not just bare "master")
    masters_patterns = [
        r"\bmaster(?:'?s)?\s+(?:degree|of|in)\b",
        r"\bmaster(?:'?s)?\s+(?:program|level|qualification)\b",
        r"\bmba\b",
        r"\bm\.?\s*s\.?\s+(?:in|degree|from)\b",
        r"\bm\.?\s*tech\b", r"\bm\.?\s*sc\b", r"\bm\.?\s*eng\b",
        r"\bpostgraduate\b",
        r"\bmaster(?:'?s)?\s+degree\b",
    ]
    if any(re.search(p, jd_lower) for p in masters_patterns):
        return "masters"
    # Bachelors: specific patterns to avoid matching bare "degree"
    bachelors_patterns = [
        r"\bbachelor(?:'?s)?\s+(?:degree|of|in)\b",
        r"\bb\.?\s*s\.?\s+(?:in|degree|from)\b",
        r"\bb\.?\s*tech\b", r"\bb\.?\s*sc\b", r"\bb\.?\s*eng\b",
        r"\bundergraduate\s+degree\b",
        r"\bbachelor\s+(?:degree|of)\b",
        # "degree in X" or "degree required" — generic but usually means bachelors
        r"\bdegree\s+(?:in|required|preferred|from)\b",
    ]
    if any(re.search(p, jd_lower) for p in bachelors_patterns):
        return "bachelors"
    if re.search(r"\b(?:diploma|associate(?:'?s)?\s+degree|certification)\b", jd_lower):
        return "diploma"
    return "none"  # JD doesn't specify education


# ---------------------------------------------------------------------------
# Experience Scoring (JD-aware)
# ---------------------------------------------------------------------------

# Seniority detection: we look for seniority QUALIFIERS as standalone words
# near role titles, rather than trying to match exact title patterns.
# This handles "Senior Software Engineer", "Lead Data Scientist",
# "Staff ML Engineer" etc. without needing to enumerate every combination.
_SENIOR_KEYWORDS = re.compile(
    r"\b(?:senior|sr\.?|lead|staff|principal|director|vp|head\s+of|chief|architect)\b",
    re.IGNORECASE,
)
_JUNIOR_KEYWORDS = re.compile(
    r"\b(?:junior|jr\.?|associate|entry[\s-]?level)\b",
    re.IGNORECASE,
)
_INTERN_KEYWORDS = re.compile(
    r"\b(?:intern|internship|trainee|apprentice|co[\s-]?op)\b",
    re.IGNORECASE,
)
_ROLE_NOUNS = re.compile(
    r"\b(?:engineer|developer|manager|analyst|scientist|designer|consultant|architect)\b",
    re.IGNORECASE,
)


def _extract_candidate_years(text: str) -> tuple[int, bool]:
    """Extract max years of experience from resume text.

    Returns:
        (years_found, was_explicitly_stated) tuple
    """
    text_lower = text.lower()
    patterns = [
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s*(?:of)?\s*(?:experience|exp)",
        r"experience\s*(?:of|:)?\s*(\d{1,2})\+?\s*(?:years?|yrs?)",
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s*(?:in|of|working)",
    ]
    years = []
    for p in patterns:
        years.extend(int(m) for m in re.findall(p, text_lower))
    if years:
        return (max(years), True)
    return (0, False)


# Seniority-to-years inference map (used when no explicit years mentioned)
_SENIORITY_YEARS_INFERENCE = {
    "senior": 8,    # Senior typically means 8+ years
    "mid": 4,       # Mid-level typically means 4-6 years
    "junior": 2,    # Junior typically means 1-3 years
    "intern": 0,    # Intern has no experience
    "unknown": 3,   # Default assumption for unknown seniority
}


def _detect_candidate_seniority(text: str) -> str:
    """Detect the highest seniority level in the resume.

    Seniority QUALIFIERS (senior, junior, intern) always take priority
    over generic role nouns. Among qualifiers, the highest wins.
    Falls back to 'mid' only when role nouns exist without any qualifier.
    """
    # Collect all qualifying levels found in text
    found_levels: list[int] = []
    if _SENIOR_KEYWORDS.search(text):
        found_levels.append(_SENIORITY_MAP["senior"])
    if _JUNIOR_KEYWORDS.search(text):
        found_levels.append(_SENIORITY_MAP["junior"])
    if _INTERN_KEYWORDS.search(text):
        found_levels.append(_SENIORITY_MAP["intern"])

    if found_levels:
        # Return the HIGHEST qualifier found
        best = max(found_levels)
        return next(k for k, v in _SENIORITY_MAP.items() if v == best)

    # No qualifiers — fall back to mid if they have role nouns
    if _ROLE_NOUNS.search(text):
        return "mid"
    return "unknown"


def score_experience(
    resume_text: str,
    jd_text: str,
    return_metadata: bool = False,
) -> float | tuple[float, dict]:
    """Score experience FIT against what the JD asks for (0-100).

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text
        return_metadata: If True, return (score, metadata) tuple with match_type info

    Returns:
        score (0-100) or (score, metadata) tuple if return_metadata=True

    Scoring logic:
    - If JD asks for 5+ years and resume has 7 → high score
    - If JD asks for senior and resume has senior roles → high score
    - If JD asks for junior but resume is senior → still decent (overqualified)
    - If JD asks for senior but resume is intern → low score

    When no explicit years are mentioned in the resume, we infer years from
    the seniority level (e.g., "Senior Engineer" → ~8 years) with reduced
    confidence weighting.
    """
    # What the JD wants
    required_years = _extract_required_years(jd_text)
    required_seniority = _extract_seniority_level(jd_text)

    # What the candidate has
    candidate_years, years_explicit = _extract_candidate_years(resume_text)
    candidate_seniority = _detect_candidate_seniority(resume_text)

    # Infer years from seniority if not explicitly stated
    years_inferred = False
    if not years_explicit and candidate_seniority != "unknown":
        candidate_years = _SENIORITY_YEARS_INFERENCE.get(candidate_seniority, 3)
        years_inferred = True

    # Date range evidence (tenure)
    date_ranges = re.findall(
        r"(?:19|20)\d{2}\s*[-\u2013]\s*(?:(?:19|20)\d{2}|present|current|now)",
        resume_text.lower(),
    )

    score = 0.0

    # --- Years match (up to 45 pts) ---
    if required_years is not None:
        if candidate_years >= required_years:
            # Full credit, but reduce if years were inferred
            pts = 45.0 if years_explicit else 35.0
            score += pts
        elif candidate_years > 0:
            # Partial credit proportional to how close they are
            ratio = candidate_years / required_years
            base_pts = 45.0 if years_explicit else 35.0
            score += round(base_pts * min(ratio, 1.0), 1)
        # else: no years mentioned → 0 pts from this dimension
    else:
        # JD doesn't specify years → give credit for any experience
        multiplier = 5 if years_explicit else 4
        score += min(candidate_years * multiplier, 30)

    # --- Seniority match (up to 35 pts) ---
    required_level = _SENIORITY_MAP.get(required_seniority, 2)
    candidate_level = _SENIORITY_MAP.get(candidate_seniority, 0)

    if candidate_level >= required_level:
        score += 35.0  # meets or exceeds
    elif candidate_level > 0:
        ratio = candidate_level / max(required_level, 1)
        score += round(35.0 * ratio, 1)

    # --- Tenure evidence (up to 20 pts) ---
    score += min(len(date_ranges) * 5, 20)

    final_score = min(round(score, 1), 100.0)

    if return_metadata:
        # Determine match type for UI display
        if candidate_level > required_level + 1:
            match_type = "overqualified"
        elif candidate_level < required_level - 1:
            match_type = "underqualified"
        else:
            match_type = "exact"

        metadata = {
            "match_type": match_type,
            "candidate_years": candidate_years,
            "years_explicit": years_explicit,
            "years_inferred": years_inferred,
            "required_years": required_years,
            "candidate_seniority": candidate_seniority,
            "required_seniority": required_seniority,
        }
        return final_score, metadata

    return final_score


# ---------------------------------------------------------------------------
# Education Scoring (JD-aware)
# ---------------------------------------------------------------------------

_DEGREE_LEVELS = {"none": 0, "diploma": 1, "bachelors": 2, "masters": 3, "phd": 4}


def _detect_candidate_education(text: str) -> str:
    """Detect highest education level in resume."""
    text_lower = text.lower()

    if re.search(r"\b(?:ph\.?d|doctorate|doctoral)\b", text_lower):
        return "phd"

    masters_patterns = [
        r"\bmaster(?:'s|s)?\s+(?:degree|of|in)\b",
        r"\bm\.?\s*s\.?\s+(?:in|degree)\b",
        r"\bm\.?\s*s\.?\b(?!\w)",
        r"\bm\.?\s*tech\b", r"\bm\.?\s*sc\b", r"\bmba\b",
        r"\bm\.?\s*eng\b", r"\bm\.?\s*a\.?\s+(?:in|degree)\b",
        r"\bpost\s*graduate|postgraduate\b",
    ]
    if any(re.search(p, text_lower) for p in masters_patterns):
        return "masters"

    bachelors_patterns = [
        r"\bbachelor(?:'s|s)?\s+(?:degree|of|in)\b",
        r"\bb\.?\s*s\.?\s+(?:in|degree)\b",
        r"\bb\.?\s*tech\b", r"\bb\.?\s*sc\b", r"\bb\.?\s*eng\b",
        r"\bb\.?\s*a\.?\s+(?:in|degree)\b",
        r"\bundergraduate\s+degree\b",
        r"\bbachelor\s+of\s+technology\b",
    ]
    if any(re.search(p, text_lower) for p in bachelors_patterns):
        return "bachelors"

    if re.search(r"\b(?:diploma|associate(?:'s)?\s+degree|certification)\b", text_lower):
        return "diploma"

    return "none"


def score_education(resume_text: str, jd_text: str) -> float:
    """Score education FIT against what the JD asks for (0-100).

    - If JD requires bachelors and resume has masters → 100
    - If JD requires masters and resume has bachelors → partial credit
    - If JD doesn't mention education → generous baseline
    """
    required = _extract_required_education(jd_text)
    candidate = _detect_candidate_education(resume_text)

    required_level = _DEGREE_LEVELS.get(required, 0)
    candidate_level = _DEGREE_LEVELS.get(candidate, 0)

    text_lower = resume_text.lower()
    score = 0.0

    if required_level == 0:
        # JD doesn't specify education → give baseline credit
        score = min(candidate_level * 20, 70)
    elif candidate_level >= required_level:
        score = 85.0  # meets or exceeds
    elif candidate_level > 0:
        ratio = candidate_level / required_level
        score = round(85.0 * ratio, 1)

    # Bonus: relevant field of study
    if re.search(
        r"\b(?:computer science|software engineering|information technology"
        r"|data science|electrical engineering|mathematics|statistics)\b",
        text_lower,
    ):
        score = min(score + 10, 100.0)

    # Bonus: honors / GPA
    if re.search(r"\b(?:gpa|cgpa|distinction|honors?|cum laude|summa)\b", text_lower):
        score = min(score + 5, 100.0)

    return round(score, 1)


# ---------------------------------------------------------------------------
# Project Scoring (JD-aware)
# ---------------------------------------------------------------------------

_PROJECT_SECTION = re.compile(
    r"(?:^|\n)\s*(?:projects?|personal projects?|side projects?"
    r"|key projects?|academic projects?|portfolio)\s*[:\n\-|]",
    re.IGNORECASE,
)
_NON_PROJECT_SECTION = re.compile(
    r"(?:^|\n)\s*(?:experience|work\s+experience|employment|professional\s+experience"
    r"|education|skills|certifications?|awards?|publications?|references?)\s*[:\n\-|]",
    re.IGNORECASE,
)


def _extract_project_section(text: str) -> str:
    """Isolate the 'Projects' section from the full resume text."""
    proj_match = _PROJECT_SECTION.search(text)
    if not proj_match:
        return text

    start = proj_match.end()
    next_section = _NON_PROJECT_SECTION.search(text, pos=start)
    end = next_section.start() if next_section else len(text)

    section_text = text[start:end].strip()
    return section_text if len(section_text) > 30 else text


def score_projects(resume_text: str, jd_text: str) -> float:
    """Score projects RELEVANCE to the JD (0-100).

    Only counts bullets/verbs from the actual Projects section.
    When no Projects section exists, gives partial credit for
    JD keyword overlap and portfolio links, but NOT for work
    experience bullets (which would falsely inflate the score).
    """
    has_section = bool(_PROJECT_SECTION.search(resume_text))
    project_text = _extract_project_section(resume_text) if has_section else ""
    proj_lower = project_text.lower()
    jd_lower = jd_text.lower()
    full_lower = resume_text.lower()

    # --- Has project section? (up to 10 pts) ---
    section_score = 10 if has_section else 0

    # --- Project bullet quality (up to 20 pts) ---
    # ONLY count bullets from the actual project section
    if has_section:
        bullets = re.findall(r"(?:^|\n)\s*[\u2022\-\*\u25cf]\s*.{20,}", proj_lower)
        bullet_score = min(len(bullets) * 4, 20)
    else:
        bullet_score = 0

    # --- Action verbs in project section (up to 15 pts) ---
    if has_section:
        action_verbs = re.findall(
            r"\b(?:built|developed|created|implemented|designed|deployed"
            r"|architected|integrated|automated|optimized|engineered)\b",
            proj_lower,
        )
        verb_score = min(len(action_verbs) * 3, 15)
    else:
        verb_score = 0

    # --- JD keyword overlap (up to 40 pts) ---
    # When project section exists, check overlap there.
    # When it doesn't, check full resume but apply a penalty (cap at 20).
    jd_keywords = set(re.findall(r"\b[a-z][a-z0-9+#.]{2,}\b", jd_lower))
    _stopwords = {
        "the", "and", "for", "with", "that", "this", "are", "was",
        "will", "from", "have", "has", "been", "our", "your", "you",
        "not", "but", "all", "can", "had", "her", "one", "who",
        "their", "there", "what", "about", "which", "when", "make",
        "like", "than", "each", "other", "into", "more", "some",
        "should", "would", "could", "must", "also", "such", "work",
        "using", "used", "including", "able", "strong", "good",
        "experience", "required", "preferred", "looking", "team",
    }
    jd_keywords -= _stopwords

    if jd_keywords:
        search_text = proj_lower if has_section else full_lower
        max_relevance = 40 if has_section else 20  # penalize no-section
        text_keywords = set(re.findall(r"\b[a-z][a-z0-9+#.]{2,}\b", search_text))
        overlap = text_keywords & jd_keywords
        relevance_ratio = len(overlap) / len(jd_keywords)
        relevance_score = round(
            max_relevance * min(relevance_ratio * 2.5, 1.0), 1,
        )
    else:
        relevance_score = 10 if has_section else 5

    # --- Portfolio/GitHub links (up to 15 pts) ---
    has_links = bool(re.search(
        r"(?:github\.com|gitlab\.com|bitbucket|portfolio|demo|heroku|vercel|netlify)",
        full_lower,
    ))
    link_score = 15 if has_links else 0

    return min(
        round(section_score + bullet_score + verb_score + relevance_score + link_score, 1),
        100.0,
    )


# ---------------------------------------------------------------------------
# Score Normalization (prevents dimension bias)
# ---------------------------------------------------------------------------

# Observed score bounds for each dimension (from empirical data)
# Skills can easily hit 100, but similarity rarely exceeds 85.
# These bounds map each dimension to a 0-100 comparable scale.
_DIMENSION_BOUNDS = {
    "skills": (0, 100),       # 0–100 naturally
    "similarity": (20, 90),   # Semantic similarity rarely > 90
    "experience": (10, 100),  # Varies based on years/seniority
    "education": (0, 100),    # 0–100 naturally
    "projects": (15, 90),     # Rarely > 90 even with great projects
}


def normalize_scores(
    scores: dict[str, float],
    enabled: bool = True,
) -> dict[str, float]:
    """Normalize section scores to a comparable 0-100 scale.

    Different sections have different natural score ranges:
    - Skills can easily hit 100% if all keywords match
    - Similarity (semantic) rarely exceeds 85-90%
    - Projects rarely exceeds 85% even with great projects

    This function maps each section's natural range to 0-100 so that
    all dimensions are comparable when computing the weighted final score.

    Args:
        scores: Dict of dimension name → raw score (0-100)
        enabled: If False, returns scores unchanged (for feature flag)

    Returns:
        Dict of dimension name → normalized score (0-100)
    """
    from app.config import FEATURE_SCORE_NORMALIZATION

    if not enabled or not FEATURE_SCORE_NORMALIZATION:
        return scores.copy()

    normalized = {}
    for dim, score in scores.items():
        bounds = _DIMENSION_BOUNDS.get(dim, (0, 100))
        low, high = bounds

        # Clamp to observed bounds
        clamped = max(low, min(score, high))

        # Map [low, high] → [0, 100]
        if high > low:
            normalized[dim] = round(((clamped - low) / (high - low)) * 100, 1)
        else:
            normalized[dim] = clamped

    return normalized
