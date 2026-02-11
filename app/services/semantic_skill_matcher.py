"""Semantic skill matching using embeddings and LLM.

Replaces regex-based skill matching with embedding-powered semantic matching.
Understands skill variations (React vs React.js), detects proficiency levels,
and identifies transferable skills.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from app.config import ENABLE_SEMANTIC_SKILLS, SEMANTIC_SKILL_THRESHOLD
from app.services.llm_service import ai_enabled, _chat_json, _parse_model
from app.services.similarity import _get_embedding
from app.services.skill_matcher import ALL_KNOWN_SKILLS, extract_skills

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models for LLM Structured Output
# ---------------------------------------------------------------------------

class ExtractedSkill(BaseModel):
    """A skill extracted from text with context."""
    name: str = Field(description="Normalized skill name (e.g., 'React' not 'ReactJS')")
    proficiency: str = Field(
        default="intermediate",
        description="One of: basic, intermediate, advanced, expert"
    )
    years: int | None = Field(
        default=None,
        description="Years of experience with this skill if mentioned"
    )
    context: str = Field(
        default="",
        description="Brief quote showing how skill was used"
    )


class ExtractedSkills(BaseModel):
    """Collection of skills extracted from a document."""
    skills: list[ExtractedSkill] = Field(default_factory=list)


class JDRequiredSkill(BaseModel):
    """A skill required by a job description."""
    name: str = Field(description="Skill name")
    importance: str = Field(
        default="required",
        description="One of: required, preferred, nice_to_have"
    )
    min_years: int | None = Field(
        default=None,
        description="Minimum years required if specified"
    )


class JDSkillRequirements(BaseModel):
    """Skills extracted from a job description."""
    skills: list[JDRequiredSkill] = Field(default_factory=list)


class PartialSkillMatch(BaseModel):
    """A partial/transferable skill match."""
    required: str
    found: str
    similarity: float
    note: str = Field(default="", description="Why this is a partial match")


class SemanticSkillResult(BaseModel):
    """Result of semantic skill matching."""
    matched: list[str] = Field(default_factory=list)
    partial: list[PartialSkillMatch] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)
    proficiency_levels: dict[str, str] = Field(default_factory=dict)
    score: float = Field(default=0.0)
    method: str = Field(default="semantic")


# ---------------------------------------------------------------------------
# Skill Embedding Cache
# ---------------------------------------------------------------------------

_SKILL_EMBEDDINGS: dict[str, list[float]] = {}


def _init_skill_embeddings() -> None:
    """Pre-compute embeddings for all skills in skill_db.json.

    Called lazily on first semantic match request.
    """
    if not ai_enabled():
        return

    if _SKILL_EMBEDDINGS:
        return  # Already initialized

    logger.info("Initializing skill embeddings for %d skills...", len(ALL_KNOWN_SKILLS))
    for skill in ALL_KNOWN_SKILLS:
        emb = _get_embedding(skill)
        if emb:
            _SKILL_EMBEDDINGS[skill] = emb
    logger.info("Cached %d skill embeddings", len(_SKILL_EMBEDDINGS))


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# LLM System Prompts
# ---------------------------------------------------------------------------

_SKILL_EXTRACTOR_SYSTEM = (
    "You are an expert technical recruiter and resume parser. "
    "Extract technical skills from text with proficiency assessment. "
    "Normalize skill names (ReactJS → React, NodeJS → Node.js). "
    "Proficiency levels based on context: "
    "- expert: led, architected, 8+ years, mentored, designed systems "
    "- advanced: implemented, optimized, 5-7 years, complex projects "
    "- intermediate: developed, built, 2-4 years, contributed "
    "- basic: familiar, exposure, coursework, learning, <2 years"
)

_JD_SKILL_EXTRACTOR_SYSTEM = (
    "You are an expert technical recruiter. "
    "Extract skill requirements from job descriptions. "
    "Classify importance: "
    "- required: 'must have', 'required', 'essential', without qualifiers "
    "- preferred: 'preferred', 'strongly preferred', 'ideally' "
    "- nice_to_have: 'nice to have', 'bonus', 'plus', 'optional'"
)


# ---------------------------------------------------------------------------
# LLM Extraction Functions
# ---------------------------------------------------------------------------

async def _extract_resume_skills_llm(resume_text: str) -> list[ExtractedSkill]:
    """Use LLM to extract skills WITH proficiency levels from resume."""

    schema = {
        "skills": [
            {
                "name": "skill_name (normalized)",
                "proficiency": "basic|intermediate|advanced|expert",
                "years": "number or null",
                "context": "brief quote showing usage"
            }
        ]
    }

    prompt = f"""Analyze this resume and extract ALL technical skills with proficiency levels.

## Resume
{resume_text[:4000]}

## Instructions
For each skill found:
1. Normalize the skill name (ReactJS → React, NodeJS → Node.js)
2. Determine proficiency from context clues
3. Extract years if mentioned
4. Note context where skill was used

Focus on technical skills, programming languages, frameworks, tools, and platforms.
"""

    data = await _chat_json(_SKILL_EXTRACTOR_SYSTEM, prompt, schema)
    if data is None:
        return []

    try:
        result = ExtractedSkills.model_validate(data)
        return result.skills
    except ValidationError:
        logger.warning("Failed to parse LLM skill extraction: %s", data)
        return []


async def _extract_jd_skills_llm(jd_text: str) -> list[JDRequiredSkill]:
    """Use LLM to extract required skills from job description."""

    schema = {
        "skills": [
            {
                "name": "skill_name",
                "importance": "required|preferred|nice_to_have",
                "min_years": "number or null"
            }
        ]
    }

    prompt = f"""Extract all technical skill requirements from this job description.

## Job Description
{jd_text[:3000]}

## Instructions
For each skill:
1. Identify the skill name
2. Classify importance (required/preferred/nice_to_have)
3. Note minimum years if specified

Include: programming languages, frameworks, tools, platforms, methodologies.
"""

    data = await _chat_json(_JD_SKILL_EXTRACTOR_SYSTEM, prompt, schema)
    if data is None:
        return []

    try:
        result = JDSkillRequirements.model_validate(data)
        return result.skills
    except ValidationError:
        logger.warning("Failed to parse LLM JD skill extraction: %s", data)
        return []


# ---------------------------------------------------------------------------
# Semantic Matching Core
# ---------------------------------------------------------------------------

async def semantic_skill_match(
    resume_text: str,
    jd_text: str,
    threshold: float | None = None,
) -> SemanticSkillResult:
    """Match skills semantically using embeddings and LLM.

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text
        threshold: Similarity threshold for partial matches (default from config)

    Returns:
        SemanticSkillResult with matched, partial, missing skills and score
    """
    if threshold is None:
        threshold = SEMANTIC_SKILL_THRESHOLD

    if not ENABLE_SEMANTIC_SKILLS or not ai_enabled():
        # Fall back to regex-based matching
        return _fallback_to_regex(resume_text, jd_text)

    # Ensure embeddings are initialized
    _init_skill_embeddings()

    # Step 1: Extract skills from both documents using LLM
    jd_skills = await _extract_jd_skills_llm(jd_text)
    resume_skills = await _extract_resume_skills_llm(resume_text)

    if not jd_skills:
        # LLM failed, fall back to regex
        logger.warning("JD skill extraction failed, falling back to regex")
        return _fallback_to_regex(resume_text, jd_text)

    # Build lookup for resume skills by normalized name
    resume_skill_map: dict[str, ExtractedSkill] = {
        s.name.lower().strip(): s for s in resume_skills
    }

    # Step 2: Match each JD requirement against resume skills
    matched: list[str] = []
    partial: list[PartialSkillMatch] = []
    missing: list[str] = []
    proficiency_levels: dict[str, str] = {}

    for jd_skill in jd_skills:
        req_name = jd_skill.name.lower().strip()

        # Direct match check
        if req_name in resume_skill_map:
            matched.append(jd_skill.name)
            proficiency_levels[jd_skill.name] = resume_skill_map[req_name].proficiency
            continue

        # Try semantic matching with embeddings
        best_match: ExtractedSkill | None = None
        best_score = 0.0

        req_emb = _get_embedding(jd_skill.name)
        if req_emb:
            for resume_skill in resume_skills:
                res_emb = _get_embedding(resume_skill.name)
                if res_emb:
                    similarity = _cosine_similarity(req_emb, res_emb)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = resume_skill

        if best_score >= 0.9:
            # Near-exact semantic match
            matched.append(jd_skill.name)
            if best_match:
                proficiency_levels[jd_skill.name] = best_match.proficiency
        elif best_score >= threshold and best_match:
            # Partial/transferable match
            partial.append(PartialSkillMatch(
                required=jd_skill.name,
                found=best_match.name,
                similarity=round(best_score, 3),
                note=f"Transferable: {best_match.name} → {jd_skill.name}"
            ))
        else:
            missing.append(jd_skill.name)

    # Step 3: Find extra skills (in resume but not required)
    required_names = {s.name.lower().strip() for s in jd_skills}
    matched_names = {s.lower() for s in matched}
    partial_found = {p.found.lower() for p in partial}

    extra: list[str] = []
    for resume_skill in resume_skills:
        skill_lower = resume_skill.name.lower().strip()
        if (skill_lower not in required_names and
            skill_lower not in matched_names and
            skill_lower not in partial_found):
            extra.append(resume_skill.name)

    # Step 4: Calculate score
    total_required = len(jd_skills) or 1

    # Weight by importance
    required_count = sum(1 for s in jd_skills if s.importance == "required")
    preferred_count = sum(1 for s in jd_skills if s.importance == "preferred")

    matched_required = sum(1 for s in jd_skills
                          if s.importance == "required" and s.name in matched)
    matched_preferred = sum(1 for s in jd_skills
                           if s.importance == "preferred" and s.name in matched)

    # Partial matches count as 0.5
    partial_credit = len(partial) * 0.5

    # Weighted calculation
    if required_count > 0:
        required_score = matched_required / required_count
    else:
        required_score = 1.0  # No required skills = full credit

    if preferred_count > 0:
        preferred_score = matched_preferred / preferred_count
    else:
        preferred_score = 1.0

    # Final score: 70% required, 20% preferred, 10% partial
    score = (
        required_score * 70 +
        preferred_score * 20 +
        (partial_credit / max(total_required, 1)) * 10
    )

    return SemanticSkillResult(
        matched=matched,
        partial=partial,
        missing=missing,
        extra=extra[:20],  # Limit extras shown
        proficiency_levels=proficiency_levels,
        score=round(min(score, 100), 1),
        method="semantic"
    )


def _fallback_to_regex(resume_text: str, jd_text: str) -> SemanticSkillResult:
    """Fall back to regex-based skill matching when LLM unavailable."""
    from app.services.skill_matcher import match_skills

    result = match_skills(resume_text, jd_text, role=None)

    return SemanticSkillResult(
        matched=result["matched"],
        partial=[],
        missing=result["missing"],
        extra=result["extra"],
        proficiency_levels={},
        score=result["score"],
        method="regex"
    )


# ---------------------------------------------------------------------------
# Proficiency-Adjusted Scoring
# ---------------------------------------------------------------------------

def adjust_score_for_proficiency(
    result: SemanticSkillResult,
    jd_skills: list[JDRequiredSkill],
) -> float:
    """Adjust skill score based on proficiency levels.

    Expert proficiency gives bonus points, basic gives penalty.
    """
    if not result.proficiency_levels:
        return result.score

    proficiency_multipliers = {
        "expert": 1.15,
        "advanced": 1.05,
        "intermediate": 1.0,
        "basic": 0.85,
    }

    adjustments = []
    for skill in result.matched:
        prof = result.proficiency_levels.get(skill, "intermediate")
        multiplier = proficiency_multipliers.get(prof, 1.0)
        adjustments.append(multiplier)

    if not adjustments:
        return result.score

    avg_multiplier = sum(adjustments) / len(adjustments)
    adjusted = result.score * avg_multiplier

    return round(min(adjusted, 100), 1)
