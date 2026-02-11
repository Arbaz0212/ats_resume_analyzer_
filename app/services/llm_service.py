"""Core LLM service powered by local Ollama.

Ollama runs entirely on your machine — no API keys, no network calls,
no data leaving your laptop. We use Ollama's native /api/chat endpoint
with JSON mode for reliable structured output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

_OLLAMA_BASE = OLLAMA_BASE_URL.replace("/v1", "")


def is_ollama_running() -> bool:
    """Quick check if Ollama is reachable."""
    try:
        resp = httpx.get(_OLLAMA_BASE, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


_ollama_available: bool | None = None


def ai_enabled() -> bool:
    """Check if AI features are available (Ollama is running)."""
    global _ollama_available
    if _ollama_available is None:
        _ollama_available = is_ollama_running()
    return _ollama_available


def reset_ollama_check() -> None:
    """Reset the cached Ollama check (useful if user starts Ollama later)."""
    global _ollama_available
    _ollama_available = None


# ---------------------------------------------------------------------------
# Core Ollama chat (JSON mode)
# ---------------------------------------------------------------------------


async def _chat_json(
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Send a chat request to Ollama with JSON output format.

    Uses Ollama's native /api/chat endpoint with format='json'
    for reliable structured output from any model.
    """
    if not ai_enabled():
        return None

    # If we have a schema, append it to the system prompt so the model
    # knows exactly what JSON structure to produce
    full_system = system_prompt
    if schema:
        full_system += (
            "\n\nYou MUST respond with valid JSON matching this exact schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{_OLLAMA_BASE}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 4096,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Ollama returned invalid JSON")
        return None
    except Exception:
        logger.exception("Ollama chat request failed")
        return None


def _parse_model[T: BaseModel](model_cls: type[T], data: dict) -> T | None:
    """Safely parse a dict into a Pydantic model.

    Handles common LLM quirks like returning strings where lists are expected.
    """
    # Fix common LLM issue: string where list[str] expected
    for field_name, field_info in model_cls.model_fields.items():
        if field_name in data and isinstance(data[field_name], str):
            origin = getattr(field_info.annotation, "__origin__", None)
            if origin is list:
                # Split string into a single-element list
                data[field_name] = [data[field_name]]
    try:
        return model_cls.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to validate %s: %s — data: %s", model_cls.__name__, exc, str(data)[:500])
        return None


# ---------------------------------------------------------------------------
# Pydantic result schemas
# ---------------------------------------------------------------------------


class JDRequirement(BaseModel):
    """A single requirement extracted from a job description."""
    skill_or_requirement: str = Field(description="The skill or requirement")
    category: str = Field(
        description="One of: technical, soft_skill, experience, education, certification",
    )
    importance: str = Field(description="One of: must_have, nice_to_have")
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative skills that satisfy this requirement",
    )


class ParsedJD(BaseModel):
    """Structured representation of a job description."""
    role_title: str = Field(description="The job title")
    seniority_level: str = Field(
        description="One of: intern, junior, mid, senior, lead, principal",
    )
    min_years_experience: int | None = Field(
        default=None, description="Minimum years required, or null",
    )
    required_education: str | None = Field(
        default=None, description="Required degree level or null",
    )
    requirements: list[JDRequirement] = Field(
        default_factory=list,
        description="All extracted requirements",
    )
    summary: str = Field(
        default="",
        description="1-2 sentence summary of what this role needs",
    )


class CandidateInsight(BaseModel):
    """AI-generated analysis of a single candidate."""
    fit_summary: str = Field(
        default="",
        description="2-3 sentence summary of how well this candidate fits",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Top 3-5 strengths relative to this JD",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Top 3-5 gaps or risks relative to this JD",
    )
    interview_questions: list[str] = Field(
        default_factory=list,
        description="3 targeted interview questions based on their gaps",
    )
    verdict: str = Field(
        default="lean_no",
        description="One of: strong_hire, lean_hire, lean_no, strong_no",
    )


class BatchSummary(BaseModel):
    """AI-generated executive summary across all candidates."""
    executive_summary: str = Field(
        default="",
        description="3-4 sentence executive summary for the hiring manager",
    )
    top_recommendation: str = Field(
        default="",
        description="Who to interview first and why (1-2 sentences)",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="3-5 specific, actionable hiring recommendations. Each should be 1-2 sentences covering: interview priorities, skill gaps to probe, team fit considerations, salary/leveling notes, and timeline urgency.",
    )
    talent_gaps: list[str] = Field(
        default_factory=list,
        description="Common skill gaps across all candidates",
    )
    hiring_risk: str = Field(
        default="medium",
        description="One of: low, medium, high based on candidate pool quality",
    )
    qualified_threshold_score: float = Field(
        default=65.0,
        description="The minimum score you'd consider 'qualified' for this specific role/JD. Consider the JD requirements strictly.",
    )


class ResumeSuggestion(BaseModel):
    """A single improvement suggestion for a resume."""
    category: str = Field(
        default="formatting",
        description="One of: missing_section, weak_skill, action_verbs, formatting, tech_stack",
    )
    current_state: str = Field(
        default="",
        description="What's currently on the resume that needs improvement",
    )
    improvement: str = Field(
        default="",
        description="Specific actionable suggestion to improve the resume",
    )
    example: str | None = Field(
        default=None,
        description="Before/after example text showing the improvement",
    )
    priority: str = Field(
        default="medium",
        description="One of: critical, high, medium, low",
    )
    jd_relevance: str = Field(
        default="",
        description="Why this improvement matters for this specific JD",
    )


class ResumeSuggestions(BaseModel):
    """AI-generated improvement suggestions for a candidate's resume."""
    total_score: float = Field(
        default=50.0,
        description="Current ATS score percentage (number only, no % sign)",
    )
    potential_score: float = Field(
        default=70.0,
        description="Estimated score if all suggestions implemented (number only, no % sign)",
    )

    @field_validator("total_score", "potential_score", mode="before")
    @classmethod
    def _parse_score(cls, v: Any) -> float:
        if isinstance(v, str):
            v = v.replace("%", "").strip().split()[0]
        return float(v)
    suggestions: list[ResumeSuggestion] = Field(
        default_factory=list,
        description="Up to 5 prioritized improvement suggestions",
    )
    summary: str = Field(
        default="",
        description="Brief 2 sentence improvement roadmap",
    )


class LLMHolisticScore(BaseModel):
    """LLM-generated holistic scoring of resume against JD."""
    requirement_fulfillment: float = Field(
        default=50.0,
        description="How well the candidate meets the stated JD requirements (0-100)",
    )
    experience_relevance: float = Field(
        default=50.0,
        description="How relevant their past experience is to this specific role (0-100)",
    )
    project_alignment: float = Field(
        default=50.0,
        description="How well their projects demonstrate needed capabilities (0-100)",
    )
    skill_depth: float = Field(
        default=50.0,
        description="Depth of skill expertise vs surface-level mention (0-100)",
    )
    overall_fit: float = Field(
        default=50.0,
        description="Overall candidate-JD fit score (0-100)",
    )
    confidence: float = Field(
        default=0.7,
        description="LLM confidence in this assessment (0-1)",
    )
    reasoning: str = Field(
        default="",
        description="2-3 sentence justification for the scores",
    )
    keyword_matches: list[str] = Field(
        default_factory=list,
        description="JD keywords found in resume",
    )
    keyword_gaps: list[str] = Field(
        default_factory=list,
        description="Important JD keywords missing from resume",
    )


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_JD_PARSER_SYSTEM = (
    "You are an expert technical recruiter and HR analyst. "
    "Parse job descriptions into structured requirements. "
    "Be precise about must_have vs nice_to_have — words like 'required', "
    "'must', 'minimum' indicate must_have. Words like 'preferred', 'bonus', "
    "'nice to have', 'ideally', 'plus' indicate nice_to_have. "
    "Detect alternative skills: 'Java or Kotlin', 'React/Angular', "
    "'AWS or Azure' — list one as the main skill and others as alternatives."
)

_CANDIDATE_ANALYST_SYSTEM = (
    "You are a senior technical hiring manager. "
    "Analyze how well a candidate's resume fits a specific job description. "
    "Be specific and actionable — reference actual skills, projects, and "
    "experience from the resume. Don't be generic. "
    "For interview questions, target the candidate's specific gaps to "
    "determine if they can grow into the role."
)

_BATCH_SUMMARY_SYSTEM = (
    "You are a VP of Engineering reviewing candidates for a specific role. "
    "Provide a thorough executive summary for the hiring manager. "
    "Be concise, data-driven, and actionable. "
    "Generate 3-5 specific recommendations covering: who to interview first, "
    "what skill gaps to probe in interviews, team composition considerations, "
    "and any urgency signals. "
    "Also identify market-level talent gaps (if all candidates lack X, that's "
    "a market signal). "
    "Set qualified_threshold_score to the minimum score you'd consider qualified "
    "for this specific JD — be strict but fair (typically 60-75 depending on role seniority)."
)

_HOLISTIC_SCORER_SYSTEM = (
    "You are an expert ATS scoring engine. Score how well a resume matches a job description. "
    "Be precise and data-driven. Reference specific skills, projects, and experience from both "
    "documents. Score each dimension 0-100 where 50 is average, 70+ is good, 85+ is excellent. "
    "Also identify keyword matches and gaps between the JD and resume."
)

_RESUME_SUGGESTIONS_SYSTEM = (
    "You are an elite ATS resume optimization consultant who has reviewed 10,000+ resumes. "
    "Analyze resumes against job descriptions and provide surgical, actionable improvements. "
    "Every suggestion MUST include a concrete before/after example using actual text from the resume. "
    "Focus on: (1) Missing keywords that ATS systems scan for, (2) Weak bullet points that can be "
    "strengthened with metrics and action verbs, (3) Skills gaps that could be bridged by reframing "
    "existing experience, (4) Section-level restructuring for ATS parsing. "
    "Each suggestion should estimate its impact on the ATS score (+1 to +5 points). "
    "Prioritize by impact: skill gaps first, then weak verbs, then formatting, then minor tweaks."
)


# ---------------------------------------------------------------------------
# Schema helpers (for JSON mode)
# ---------------------------------------------------------------------------

def _schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Generate a simplified JSON schema for the LLM prompt."""
    schema = model_cls.model_json_schema()
    # Remove $defs and other metadata that confuse local models
    simplified = {}
    for field_name, field_info in model_cls.model_fields.items():
        desc = field_info.description or ""
        simplified[field_name] = desc
    return simplified


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------


async def parse_jd(jd_text: str) -> ParsedJD | None:
    """Use LLM to extract structured requirements from a JD."""
    schema = _schema_for(ParsedJD)
    data = await _chat_json(
        system_prompt=_JD_PARSER_SYSTEM,
        user_prompt=(
            "Parse this job description into structured requirements:\n\n"
            + jd_text[:4000]
        ),
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(ParsedJD, data)


async def analyze_candidate(
    resume_text: str,
    jd_text: str,
    score_data: dict[str, Any],
) -> CandidateInsight | None:
    """Use LLM to generate deep insights for a single candidate."""
    schema = _schema_for(CandidateInsight)
    prompt = (
        f"Analyze this candidate for the role.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text[:2000]}\n\n"
        f"--- RESUME ---\n{resume_text[:3000]}\n\n"
        f"--- SCORING DATA ---\n"
        f"Overall score: {score_data.get('final_score')}%\n"
        f"Matched skills: {', '.join(score_data.get('matched_skills', []))}\n"
        f"Missing skills: {', '.join(score_data.get('missing_skills', []))}\n"
        f"Section scores: {score_data.get('section_scores')}\n"
    )
    data = await _chat_json(
        system_prompt=_CANDIDATE_ANALYST_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(CandidateInsight, data)


async def generate_batch_summary(
    jd_text: str,
    candidates: list[dict[str, Any]],
) -> BatchSummary | None:
    """Use LLM to generate an executive summary across all candidates."""
    schema = _schema_for(BatchSummary)
    candidate_lines = []
    for i, c in enumerate(candidates[:15], 1):
        candidate_lines.append(
            f"{i}. {c['candidate']} \u2014 Score: {c['final_score']}%, "
            f"Decision: {c['decision']}, "
            f"Skills: {c['skill_match_ratio']}, "
            f"Matched: {', '.join(c.get('matched_skills', [])[:8])}"
        )

    prompt = (
        f"Generate an executive hiring summary.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text[:2000]}\n\n"
        f"--- CANDIDATES (ranked by score) ---\n"
        + "\n".join(candidate_lines)
    )
    data = await _chat_json(
        system_prompt=_BATCH_SUMMARY_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(BatchSummary, data)


async def generate_resume_suggestions(
    resume_text: str,
    jd_text: str,
    score_data: dict[str, Any],
) -> ResumeSuggestions | None:
    """Use LLM to generate specific resume improvement suggestions.

    Returns up to 5 prioritized suggestions with actionable examples.
    Gracefully returns None if Ollama is unavailable or times out.
    """
    from app.config import FEATURE_IMPROVEMENT_SUGGESTIONS

    if not FEATURE_IMPROVEMENT_SUGGESTIONS:
        return None

    schema = _schema_for(ResumeSuggestions)
    total_score = score_data.get("final_score", 0)
    skills_score = score_data.get("section_scores", {}).get("skills", 0)
    experience_score = score_data.get("section_scores", {}).get("experience", 0)
    similarity_score = score_data.get("section_scores", {}).get("similarity", 0)

    prompt = f"""Analyze this resume against the job description and provide exactly 5 high-impact improvement suggestions.

--- JOB DESCRIPTION ---
{jd_text[:2500]}

--- RESUME ---
{resume_text[:3500]}

--- CURRENT SCORES ---
- Overall ATS Score: {total_score}%
- Skills Match: {skills_score}%
- Experience Match: {experience_score}%
- Similarity Score: {similarity_score}%
- Matched Skills: {', '.join(score_data.get('matched_skills', [])[:10])}
- Missing Skills: {', '.join(score_data.get('missing_skills', [])[:10])}

## INSTRUCTIONS

For EACH of the 5 suggestions, you MUST provide:

1. **category**: One of: missing_section, weak_skill, action_verbs, formatting, tech_stack
2. **current_state**: Quote the EXACT text from the resume that needs improvement (or describe what's missing)
3. **improvement**: The specific fix — what to write, where to add it, how to reword it
4. **example**: A before/after pair showing the transformation:
   - BEFORE: "Worked on backend systems using Python"
   - AFTER: "Architected and deployed 3 microservices using Python/FastAPI, reducing API latency by 40% and handling 10K+ RPM"
5. **priority**: critical (missing must-have skill), high (weak experience framing), medium (formatting), low (minor tweak)
6. **jd_relevance**: Explain WHY this change matters for THIS specific JD — reference a specific requirement

## SCORING ESTIMATION
- Estimate potential_score if all 5 suggestions are implemented (be realistic: usually +8-20 points)
- In the summary, provide a 2-sentence improvement roadmap prioritized by impact

Focus on changes that will make the BIGGEST difference in ATS keyword matching and recruiter impression."""

    data = await _chat_json(
        system_prompt=_RESUME_SUGGESTIONS_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None

    # Ensure total_score is set from actual score data
    if data.get("total_score") is None or data.get("total_score") == 0:
        data["total_score"] = total_score

    return _parse_model(ResumeSuggestions, data)


async def score_resume_with_llm(
    resume_text: str,
    jd_text: str,
) -> LLMHolisticScore | None:
    """Use LLM to generate a holistic score of resume against JD."""
    if not ai_enabled():
        return None

    schema = _schema_for(LLMHolisticScore)
    prompt = f"""Score this resume against the job description across multiple dimensions.

--- JOB DESCRIPTION ---
{jd_text[:3000]}

--- RESUME ---
{resume_text[:3000]}

For each dimension, provide a score 0-100:
1. REQUIREMENT FULFILLMENT: Does the candidate meet the stated requirements?
2. EXPERIENCE RELEVANCE: Is their experience directly applicable?
3. PROJECT ALIGNMENT: Do their projects show the right capabilities?
4. SKILL DEPTH: Do they show deep expertise or just keyword-stuffing?
5. OVERALL FIT: Weighted combination considering the role's priorities.

Also extract:
- keyword_matches: JD keywords/phrases that appear in the resume
- keyword_gaps: Important JD keywords NOT found in the resume

Provide a confidence score (0-1) for your assessment.
Be specific in your reasoning - cite evidence from both documents."""

    data = await _chat_json(
        system_prompt=_HOLISTIC_SCORER_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(LLMHolisticScore, data)


# ---------------------------------------------------------------------------
# Phase 3: Career Trajectory Analysis
# ---------------------------------------------------------------------------

class CareerTrajectory(BaseModel):
    """Analysis of career progression patterns."""
    progression_type: str = Field(
        default="mixed",
        description="One of: rapid_growth, steady_growth, lateral, stagnant, declining, pivot",
    )
    average_tenure_months: int = Field(
        default=24,
        description="Average time at each company",
    )

    @field_validator("average_tenure_months", mode="before")
    @classmethod
    def _parse_tenure(cls, v: Any) -> int:
        if isinstance(v, str):
            # Extract first number from strings like "29 months (TechCorp: 48...)"
            import re
            match = re.search(r'\d+', v)
            if match:
                return int(match.group())
        return int(v)

    job_hopping_risk: str = Field(
        default="medium",
        description="One of: low, medium, high based on tenure patterns",
    )
    employment_gaps: list[dict] = Field(
        default_factory=list,
        description="List of gaps: {start, end, months, explanation_found}",
    )
    title_progression: list[str] = Field(
        default_factory=list,
        description="Sequence of job titles",
    )
    company_tier_progression: str = Field(
        default="consistent",
        description="One of: startup_to_enterprise, enterprise_to_startup, consistent, varied",
    )
    pivot_detected: bool = Field(
        default=False,
        description="True if career pivot detected",
    )
    pivot_details: str | None = Field(
        default=None,
        description="Details of career pivot if detected",
    )
    growth_potential: str = Field(
        default="medium",
        description="One of: high, medium, low based on trajectory",
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Career red flags detected",
    )
    green_flags: list[str] = Field(
        default_factory=list,
        description="Positive career indicators",
    )


_CAREER_TRAJECTORY_SYSTEM = (
    "You are a senior technical recruiter with 15+ years experience. "
    "Analyze career progression patterns from resumes. "
    "Look for growth signals, red flags, and trajectory indicators. "
    "Context matters - short tenures at startups are different from enterprises."
)


async def analyze_career_trajectory(
    resume_text: str,
) -> CareerTrajectory | None:
    """Analyze career progression patterns and detect red flags.

    Args:
        resume_text: The candidate's resume text

    Returns:
        CareerTrajectory with progression analysis
    """
    from app.config import ENABLE_CAREER_TRAJECTORY

    if not ENABLE_CAREER_TRAJECTORY or not ai_enabled():
        return None

    schema = _schema_for(CareerTrajectory)

    prompt = f"""Analyze this resume for career trajectory patterns.

## Resume
{resume_text[:4000]}

## Analysis Required

### 1. PROGRESSION TYPE
- rapid_growth: Quick promotions, increasing responsibility faster than normal
- steady_growth: Normal 2-3 year cycles with promotions
- lateral: Same-level moves between companies
- stagnant: Same role for 5+ years without growth
- declining: Decreasing responsibility over time
- pivot: Career change to different field

### 2. JOB HOPPING ANALYSIS
- Average tenure at each company
- Pattern: Are short tenures at startups (acceptable) or enterprises (concerning)?
- Risk assessment: low (>3yr avg), medium (2-3yr), high (<2yr)

### 3. EMPLOYMENT GAPS
- Identify any gaps > 3 months between roles
- Note if gap is explained (education, travel, family, personal projects)
- Unexplained gaps are yellow flags

### 4. TITLE PROGRESSION
- Extract the sequence of job titles from resume

### 5. COMPANY TIER PROGRESSION
- startup_to_enterprise: Growth signal (skills validated by larger org)
- enterprise_to_startup: Could be positive (entrepreneurial) or negative (pushed out)
- consistent: Same tier throughout
- varied: Mix of company sizes

### 6. PIVOT DETECTION
- Has the candidate changed career direction significantly?
- If yes, what was the pivot and when?

### 7. GROWTH POTENTIAL
- high: Strong upward trajectory, learning new skills
- medium: Steady progression
- low: Stagnant or declining

### 8. RED FLAGS
- Multiple short tenures (<1 year) at enterprises
- Decreasing seniority over time
- Large unexplained gaps
- Title doesn't match responsibilities
- Same role for 7+ years without promotion

### 9. GREEN FLAGS
- Promotions within same company
- Increasing company prestige over time
- Consistent growth trajectory
- Rehired by former employer
- Long tenures at strong companies

Respond with valid JSON only."""

    data = await _chat_json(
        system_prompt=_CAREER_TRAJECTORY_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(CareerTrajectory, data)


# ---------------------------------------------------------------------------
# Phase 3: Comparative Candidate Ranking
# ---------------------------------------------------------------------------

class CandidateRanking(BaseModel):
    """Ranking entry for a single candidate."""
    candidate: str = Field(description="Candidate filename")
    rank: int = Field(description="Rank position (1 = best)")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class CandidateComparison(BaseModel):
    """Comparative analysis between candidates."""
    rankings: list[CandidateRanking] = Field(
        default_factory=list,
        description="Candidates ranked with justification",
    )
    best_for_technical: str = Field(
        default="",
        description="Candidate name strongest technically",
    )
    best_for_experience: str = Field(
        default="",
        description="Candidate with best experience match",
    )
    best_for_culture: str = Field(
        default="",
        description="Best culture fit indicators",
    )
    best_overall: str = Field(
        default="",
        description="Recommended hire",
    )
    stack_rank_justification: str = Field(
        default="",
        description="Why this ranking order",
    )
    hiring_recommendation: str = Field(
        default="need_more_candidates",
        description="One of: hire_top_1, hire_top_2, hire_none, need_more_candidates",
    )
    differentiation_factors: list[str] = Field(
        default_factory=list,
        description="Key factors that differentiate the candidates",
    )


_CANDIDATE_COMPARISON_SYSTEM = (
    "You are a VP of Engineering making final hiring decisions. "
    "Compare candidates against EACH OTHER, not just against the JD. "
    "Identify relative strengths and weaknesses between candidates. "
    "Provide actionable hiring recommendations."
)


async def compare_candidates(
    candidates: list[dict[str, Any]],
    jd_text: str,
) -> CandidateComparison | None:
    """Compare all candidates against each other for stack ranking.

    Args:
        candidates: List of scored candidates with their data
        jd_text: The job description text

    Returns:
        CandidateComparison with stack ranking and recommendations
    """
    from app.config import ENABLE_COMPARATIVE_RANKING

    if not ENABLE_COMPARATIVE_RANKING or not ai_enabled():
        return None

    if len(candidates) < 2:
        return None  # Need at least 2 candidates to compare

    schema = _schema_for(CandidateComparison)

    # Build candidate summaries
    summaries = []
    for c in candidates[:10]:  # Limit to top 10 for LLM context
        summaries.append(f"""
**{c['candidate']}** (Score: {c['final_score']}%)
- Decision: {c['decision']}
- Skills: {c['skill_match_ratio']} matched
- Matched: {', '.join(c.get('matched_skills', [])[:8])}
- Missing: {', '.join(c.get('missing_skills', [])[:5])}
- Section Scores: {c.get('section_scores', {})}""")

    prompt = f"""Compare these candidates for the role and provide stack ranking.

## Job Description
{jd_text[:1500]}

## Candidates
{"".join(summaries)}

## Analysis Required

### 1. STACK RANKING
Rank ALL candidates from best to worst with specific justification.
For each candidate explain:
- Why they're ranked at this position
- Their key strengths RELATIVE TO OTHER CANDIDATES
- Their key weaknesses RELATIVE TO OTHER CANDIDATES

### 2. DIMENSION WINNERS
- Who is STRONGEST technically? Why?
- Who has the BEST experience match? Why?
- Who shows the BEST culture fit indicators? Why?

### 3. HIRING RECOMMENDATION
- hire_top_1: Clear top candidate, strongly recommend
- hire_top_2: Top 2 are close, either would be good
- hire_none: No candidates meet the bar for this role
- need_more_candidates: Pool too weak or small

### 4. DIFFERENTIATION FACTORS
What are the key factors that separate these candidates?
(e.g., "Candidate A has production Kubernetes experience that B lacks")

Respond with valid JSON only."""

    data = await _chat_json(
        system_prompt=_CANDIDATE_COMPARISON_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(CandidateComparison, data)


# ---------------------------------------------------------------------------
# Phase 4: LLM Holistic Scoring
# ---------------------------------------------------------------------------

class LLMHolisticScore(BaseModel):
    """LLM-generated holistic scoring of resume against JD."""
    requirement_fulfillment: float = Field(
        default=50.0,
        description="How well the candidate meets the stated JD requirements (0-100)",
    )
    experience_relevance: float = Field(
        default=50.0,
        description="How relevant their past experience is to this specific role (0-100)",
    )
    project_alignment: float = Field(
        default=50.0,
        description="How well their projects demonstrate needed capabilities (0-100)",
    )
    skill_depth: float = Field(
        default=50.0,
        description="Depth of skill expertise vs surface-level mention (0-100)",
    )
    overall_fit: float = Field(
        default=50.0,
        description="Overall candidate-JD fit score (0-100)",
    )
    confidence: float = Field(
        default=0.7,
        description="LLM confidence in this assessment (0-1)",
    )
    reasoning: str = Field(
        default="",
        description="2-3 sentence justification for the scores",
    )
    keyword_matches: list[str] = Field(
        default_factory=list,
        description="JD keywords found in resume",
    )
    keyword_gaps: list[str] = Field(
        default_factory=list,
        description="Important JD keywords missing from resume",
    )


_HOLISTIC_SCORER_SYSTEM = (
    "You are an expert ATS scoring engine. Score how well a resume matches a job description. "
    "Be precise and data-driven. Reference specific skills, projects, and experience from both "
    "documents. Score each dimension 0-100 where 50 is average, 70+ is good, 85+ is excellent. "
    "Also identify keyword matches and gaps between the JD and resume."
)


async def score_resume_with_llm(
    resume_text: str,
    jd_text: str,
) -> LLMHolisticScore | None:
    """Use LLM to generate a holistic score of resume against JD."""
    if not ai_enabled():
        return None

    schema = _schema_for(LLMHolisticScore)
    prompt = f"""Score this resume against the job description across multiple dimensions.

--- JOB DESCRIPTION ---
{jd_text[:3000]}

--- RESUME ---
{resume_text[:3000]}

For each dimension, provide a score 0-100:
1. REQUIREMENT FULFILLMENT: Does the candidate meet the stated requirements?
2. EXPERIENCE RELEVANCE: Is their experience directly applicable?
3. PROJECT ALIGNMENT: Do their projects show the right capabilities?
4. SKILL DEPTH: Do they show deep expertise or just keyword-stuffing?
5. OVERALL FIT: Weighted combination considering the role's priorities.

Also extract:
- keyword_matches: JD keywords/phrases that appear in the resume
- keyword_gaps: Important JD keywords NOT found in the resume

Provide a confidence score (0-1) for your assessment.
Be specific in your reasoning - cite evidence from both documents."""

    data = await _chat_json(
        system_prompt=_HOLISTIC_SCORER_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(LLMHolisticScore, data)
