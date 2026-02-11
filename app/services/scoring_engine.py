"""Central ATS scoring orchestrator with LLM integration.

Every dimension scores a resume AGAINST the JD.
When Ollama is running locally, AI provides:
  - Semantic similarity (embeddings via nomic-embed-text)
  - Per-candidate deep insights (strengths, gaps, interview Qs)
  - Executive hiring summary
  - Resume improvement suggestions (per candidate)
  - Semantic skill matching (Phase 3)
  - Context-aware experience analysis (Phase 3)
  - Achievement impact analysis (Phase 3)
  - Multi-dimensional fit scoring (Phase 3)
  - Career trajectory analysis (Phase 3)
  - Red flag detection (Phase 3)
  - Comparative candidate ranking (Phase 3)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.skill_matcher import match_skills, get_role_weights
from app.services.similarity import calculate_similarity
from app.services.section_scorer import (
    score_experience,
    score_education,
    score_projects,
    normalize_scores,
)
from app.services.llm_service import (
    ai_enabled,
    analyze_candidate,
    generate_batch_summary,
    generate_resume_suggestions,
    analyze_career_trajectory,
    compare_candidates,
)
from app.services.llm_service import score_resume_with_llm

# Phase 3 imports - LLM-powered analysis
from app.services.semantic_skill_matcher import semantic_skill_match
from app.services.experience_analyzer import (
    analyze_experience_relevance,
    get_experience_metadata,
)
from app.services.achievement_analyzer import (
    analyze_achievements,
    get_achievement_bonus,
)
from app.services.fit_analyzer import (
    analyze_candidate_fit,
    get_fit_modifier,
)
from app.services.red_flag_detector import (
    detect_red_flags,
    get_red_flag_penalty,
)

logger = logging.getLogger(__name__)


def _hiring_decision(score: float) -> tuple[str, str]:
    """Return (label, color) for a hiring recommendation."""
    if score >= 80:
        return "Strong Match", "green"
    if score >= 65:
        return "Potential Match", "blue"
    if score >= 50:
        return "Needs Review", "amber"
    return "Weak Match", "red"


def score_resume(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
    filename: str = "unknown",
) -> dict[str, Any]:
    """Score a single resume against a JD.

    Role is optional and used for advisory context only.
    Scoring uses unified weights regardless of role selection.
    """
    skill_result = match_skills(resume_text, jd_text, role)
    similarity, sim_method = calculate_similarity(resume_text, jd_text)
    experience, exp_metadata = score_experience(resume_text, jd_text, return_metadata=True)
    education = score_education(resume_text, jd_text)
    projects = score_projects(resume_text, jd_text)

    # Raw section scores
    raw_scores = {
        "skills": skill_result["score"],
        "similarity": similarity,
        "experience": experience,
        "education": education,
        "projects": projects,
    }

    # Apply normalization to prevent dimension bias
    normalized = normalize_scores(raw_scores)

    weights = get_role_weights(role)
    total_weight = sum(weights.values())

    # Compute final score using normalized values
    final_score = round(
        (
            normalized["skills"] * weights["skills"]
            + normalized["similarity"] * weights["similarity"]
            + normalized["experience"] * weights["experience"]
            + normalized["projects"] * weights["projects"]
            + normalized["education"] * weights["education"]
        )
        / total_weight,
        1,
    )

    decision, color = _hiring_decision(final_score)

    return {
        "candidate": filename,
        "final_score": final_score,
        "decision": decision,
        "decision_color": color,
        "matched_skills": skill_result["matched"],
        "missing_skills": skill_result["missing"],
        "extra_skills": skill_result["extra"],
        "section_scores": raw_scores,  # Keep raw scores for display
        "section_scores_normalized": normalized,  # Include normalized for transparency
        "similarity_method": sim_method,
        "weights": weights,
        "skill_match_ratio": f"{skill_result['matched_count']}/{skill_result['total_required']}",
        # Experience metadata for UI indicators
        "match_type": exp_metadata.get("match_type", "exact"),
        "experience_metadata": exp_metadata,
        # Placeholder for AI insights (populated async later)
        "ai_insight": None,
        # Placeholder for improvement suggestions (populated async later)
        "improvement_suggestions": None,
        # Phase 3: Placeholders for LLM-powered analysis
        "semantic_skills": None,
        "experience_analysis": None,
        "achievement_analysis": None,
        "career_trajectory": None,
        "candidate_fit": None,
        "red_flags": None,
        "llm_holistic_score": None,
        # Store raw text for AI analysis
        "_resume_text": resume_text,
    }


async def score_resume_enhanced(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
    filename: str = "unknown",
) -> dict[str, Any]:
    """Enhanced scoring with full LLM integration when available.

    This function runs the deterministic scoring first, then enhances
    with LLM-powered analysis running in parallel for speed.
    """
    # Phase 1: Start with deterministic scoring
    base_result = score_resume(resume_text, jd_text, role, filename)

    if not ai_enabled():
        return base_result

    # Phase 2: Run LLM analyses in parallel
    try:
        (
            semantic_skills_result,
            experience_analysis,
            achievement_analysis,
            career_trajectory,
            candidate_fit,
            red_flags,
            llm_holistic_score,
        ) = await asyncio.gather(
            semantic_skill_match(resume_text, jd_text),
            analyze_experience_relevance(resume_text, jd_text),
            analyze_achievements(resume_text, jd_text),
            analyze_career_trajectory(resume_text),
            analyze_candidate_fit(resume_text, jd_text, role),
            detect_red_flags(resume_text, jd_text),
            score_resume_with_llm(resume_text, jd_text),
            return_exceptions=True,
        )

        # Handle exceptions gracefully
        if isinstance(semantic_skills_result, Exception):
            logger.warning("Semantic skills failed: %s", semantic_skills_result)
            semantic_skills_result = None
        if isinstance(experience_analysis, Exception):
            logger.warning("Experience analysis failed: %s", experience_analysis)
            experience_analysis = None
        if isinstance(achievement_analysis, Exception):
            logger.warning("Achievement analysis failed: %s", achievement_analysis)
            achievement_analysis = None
        if isinstance(career_trajectory, Exception):
            logger.warning("Career trajectory failed: %s", career_trajectory)
            career_trajectory = None
        if isinstance(candidate_fit, Exception):
            logger.warning("Candidate fit failed: %s", candidate_fit)
            candidate_fit = None
        if isinstance(red_flags, Exception):
            logger.warning("Red flags failed: %s", red_flags)
            red_flags = None
        if isinstance(llm_holistic_score, Exception):
            logger.warning("LLM holistic scoring failed: %s", llm_holistic_score)
            llm_holistic_score = None

        # Phase 3: Enhance the base score with LLM insights
        final_score = base_result["final_score"]

        # Apply modifiers based on LLM analysis
        if achievement_analysis:
            achievement_bonus = get_achievement_bonus(achievement_analysis)
            final_score += achievement_bonus

        if candidate_fit:
            fit_modifier = get_fit_modifier(candidate_fit)
            final_score += fit_modifier

        if red_flags:
            red_flag_penalty = get_red_flag_penalty(red_flags)
            final_score += red_flag_penalty

        # Blend LLM holistic score (when available)
        if llm_holistic_score and hasattr(llm_holistic_score, 'overall_fit'):
            llm_score = llm_holistic_score.overall_fit
            # Weight: 40% LLM holistic, 60% deterministic+modifiers
            final_score = 0.6 * final_score + 0.4 * llm_score

        # Clamp score to 0-100
        final_score = max(0, min(100, round(final_score, 1)))

        # Update experience metadata if LLM analysis available
        if experience_analysis:
            base_result["experience_metadata"] = get_experience_metadata(experience_analysis)
            base_result["match_type"] = base_result["experience_metadata"].get("match_type", "exact")

        # Store LLM analysis results
        base_result["final_score"] = final_score
        base_result["semantic_skills"] = (
            semantic_skills_result.model_dump()
            if semantic_skills_result and hasattr(semantic_skills_result, 'model_dump')
            else semantic_skills_result
        )
        base_result["experience_analysis"] = (
            experience_analysis.model_dump()
            if experience_analysis and hasattr(experience_analysis, 'model_dump')
            else None
        )
        base_result["achievement_analysis"] = (
            achievement_analysis.model_dump()
            if achievement_analysis and hasattr(achievement_analysis, 'model_dump')
            else None
        )
        base_result["career_trajectory"] = (
            career_trajectory.model_dump()
            if career_trajectory and hasattr(career_trajectory, 'model_dump')
            else None
        )
        base_result["candidate_fit"] = (
            candidate_fit.model_dump()
            if candidate_fit and hasattr(candidate_fit, 'model_dump')
            else None
        )
        base_result["red_flags"] = (
            red_flags.model_dump()
            if red_flags and hasattr(red_flags, 'model_dump')
            else None
        )
        base_result["llm_holistic_score"] = (
            llm_holistic_score.model_dump()
            if llm_holistic_score and hasattr(llm_holistic_score, 'model_dump')
            else None
        )

        # Update decision based on new score
        decision, color = _hiring_decision(final_score)
        base_result["decision"] = decision
        base_result["decision_color"] = color

    except Exception:
        logger.exception("Enhanced scoring failed (non-fatal), using base score")

    return base_result


async def analyze_batch(
    resumes: list[dict[str, str]],
    jd_text: str,
    role: str | None = None,
    use_enhanced_scoring: bool = True,
) -> dict[str, Any]:
    """Analyze a batch of resumes with optional AI insights and suggestions.

    Role is optional and used for advisory context only.
    Scoring uses unified weights regardless of role selection.

    Args:
        resumes: List of dicts with 'text' and 'filename' keys
        jd_text: The job description text
        role: Optional role name for context
        use_enhanced_scoring: If True and AI enabled, use LLM-enhanced scoring
    """
    # Phase 1: Score all resumes (enhanced if AI available and enabled)
    if use_enhanced_scoring and ai_enabled():
        # Use enhanced scoring with parallel LLM analysis
        scoring_tasks = [
            score_resume_enhanced(r["text"], jd_text, role, r["filename"])
            for r in resumes
        ]
        results = await asyncio.gather(*scoring_tasks, return_exceptions=True)

        # Handle any scoring failures
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Enhanced scoring failed for %s: %s", resumes[i]["filename"], result)
                # Fall back to basic scoring
                valid_results.append(score_resume(resumes[i]["text"], jd_text, role, resumes[i]["filename"]))
            else:
                valid_results.append(result)
        results = valid_results
    else:
        # Use basic deterministic scoring
        results = [
            score_resume(r["text"], jd_text, role, r["filename"])
            for r in resumes
        ]

    ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # Phase 2: AI analysis (async, only if enabled)
    ai_summary = None
    comparative_analysis = None

    if ai_enabled():
        try:
            # Run AI candidate analysis concurrently
            insight_tasks = [
                analyze_candidate(c["_resume_text"], jd_text, c)
                for c in ranked
            ]
            insights = await asyncio.gather(*insight_tasks, return_exceptions=True)
            for c, insight in zip(ranked, insights):
                if isinstance(insight, Exception):
                    logger.warning("AI insight failed for %s: %s", c["candidate"], insight)
                else:
                    c["ai_insight"] = insight.model_dump() if insight else None

            # Run improvement suggestions concurrently
            suggestion_tasks = [
                generate_resume_suggestions(c["_resume_text"], jd_text, c)
                for c in ranked
            ]
            suggestions = await asyncio.gather(*suggestion_tasks, return_exceptions=True)
            for c, sugg in zip(ranked, suggestions):
                if isinstance(sugg, Exception):
                    logger.warning("Suggestions failed for %s: %s", c["candidate"], sugg)
                else:
                    c["improvement_suggestions"] = sugg.model_dump() if sugg else None

            # Generate executive summary
            ai_summary_obj = await generate_batch_summary(jd_text, ranked)
            if ai_summary_obj:
                ai_summary = ai_summary_obj.model_dump()

            # Phase 3: Comparative candidate analysis (if 2+ candidates)
            if len(ranked) >= 2:
                try:
                    comparison = await compare_candidates(ranked, jd_text)
                    if comparison:
                        comparative_analysis = comparison.model_dump()
                except Exception:
                    logger.exception("Comparative analysis failed (non-fatal)")

        except Exception:
            logger.exception("AI batch analysis failed (non-fatal)")

    # Clean up internal fields
    for c in ranked:
        c.pop("_resume_text", None)

    return {
        "job_role": role or "General",
        "total_resumes": len(resumes),
        "all_candidates": ranked,
        "ai_enabled": ai_enabled(),
        "ai_summary": ai_summary,
        "comparative_analysis": comparative_analysis,
    }


async def analyze_single_resume(
    resume_text: str,
    jd_text: str | None = None,
    role: str | None = None,
    filename: str = "resume",
) -> dict[str, Any]:
    """Analyze a single resume for the Individual Resume Analyzer.

    Unlike analyze_batch (recruiter view), this focuses on giving
    the job seeker personal feedback. JD is optional — without it,
    we provide general resume quality feedback.
    """
    # Use a generic JD placeholder if none provided
    has_jd = bool(jd_text and jd_text.strip())
    if not has_jd:
        # General quality assessment — use a broad "any role" JD
        jd_text = (
            "Looking for a skilled professional with strong technical abilities, "
            "relevant work experience, solid educational background, and "
            "demonstrated project experience. Must have clear communication skills "
            "and a track record of delivering results."
        )

    # Run enhanced scoring (with LLM if available)
    if ai_enabled():
        result = await score_resume_enhanced(resume_text, jd_text, role, filename)
    else:
        result = score_resume(resume_text, jd_text, role, filename)

    # Run AI insights if available
    if ai_enabled():
        try:
            # Run candidate insight + suggestions in parallel
            insight_task = analyze_candidate(resume_text, jd_text, result)
            suggestion_task = generate_resume_suggestions(resume_text, jd_text, result)

            insight, suggestions = await asyncio.gather(
                insight_task, suggestion_task, return_exceptions=True,
            )

            if not isinstance(insight, Exception) and insight:
                result["ai_insight"] = insight.model_dump()
            if not isinstance(suggestions, Exception) and suggestions:
                result["improvement_suggestions"] = suggestions.model_dump()
        except Exception:
            logger.exception("AI analysis failed for single resume (non-fatal)")

    # Clean up internal fields
    result.pop("_resume_text", None)

    return result
