"""Skill extraction, implication expansion, and matching against role profiles."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(filename: str) -> Any:
    with open(_DATA_DIR / filename, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Data loaded once at import time
# ---------------------------------------------------------------------------
_RAW_SKILL_DB: dict[str, list[str]] = _load_json("skill_db.json")
ROLE_PROFILES: dict[str, dict] = _load_json("role_profiles.json")

# Flatten all known skills into a lookup set
ALL_KNOWN_SKILLS: set[str] = {
    skill.strip().lower()
    for category_skills in _RAW_SKILL_DB.values()
    for skill in category_skills
}

# ---------------------------------------------------------------------------
# Skill Implication Map
# If someone knows skill X, they implicitly know skills Y, Z, ...
# This models real-world knowledge transfer between related technologies.
# ---------------------------------------------------------------------------
SKILL_IMPLIES: dict[str, list[str]] = {
    # Languages: advanced implies base
    "c++":          ["c"],
    "c#":           ["c"],
    "typescript":   ["javascript"],
    "kotlin":       ["java"],
    "scala":        ["java"],
    "objective-c":  ["c"],
    "dart":         ["javascript"],

    # Frontend frameworks imply core web + JS
    "react":        ["javascript", "html", "css"],
    "angular":      ["javascript", "typescript", "html", "css"],
    "vue":          ["javascript", "html", "css"],
    "svelte":       ["javascript", "html", "css"],
    "next.js":      ["react", "javascript", "html", "css"],
    "nuxt.js":      ["vue", "javascript", "html", "css"],
    "gatsby":       ["react", "javascript"],
    "react native": ["react", "javascript"],

    # Backend frameworks imply their language
    "django":       ["python"],
    "flask":        ["python"],
    "fastapi":      ["python"],
    "spring boot":  ["java", "spring"],
    "spring":       ["java"],
    "express":      ["node.js", "javascript"],
    "node.js":      ["javascript"],
    "rails":        ["ruby"],
    "laravel":      ["php"],
    "asp.net":      [".net", "c#"],
    ".net":         ["c#"],

    # DevOps: orchestration implies containerization
    "kubernetes":   ["docker"],
    "helm":         ["kubernetes", "docker"],
    "ecs":          ["docker", "aws"],
    "eks":          ["kubernetes", "docker", "aws"],

    # Cloud services imply their platform
    "lambda":       ["aws", "serverless"],
    "s3":           ["aws"],
    "ec2":          ["aws"],
    "cloud functions": ["gcp", "serverless"],
    "cloud run":    ["gcp"],
    "app engine":   ["gcp"],
    "bigquery":     ["gcp", "sql"],

    # Data science: frameworks imply language
    "pandas":       ["python"],
    "numpy":        ["python"],
    "scipy":        ["python"],
    "scikit-learn": ["python", "machine learning"],
    "tensorflow":   ["python", "deep learning", "machine learning"],
    "pytorch":      ["python", "deep learning", "machine learning"],
    "keras":        ["python", "deep learning"],
    "xgboost":      ["python", "machine learning"],
    "lightgbm":     ["python", "machine learning"],

    # Data engineering
    "airflow":      ["python"],
    "dbt":          ["sql"],
    "spark":        ["python", "sql"],
    "databricks":   ["spark", "python"],

    # Testing frameworks imply language
    "pytest":       ["python"],
    "jest":         ["javascript"],
    "junit":        ["java"],
    "testng":       ["java"],
    "cypress":      ["javascript"],
    "playwright":   ["javascript"],
    "selenium":     ["automation testing"],

    # Mobile implies platform knowledge
    "flutter":      ["dart", "mobile development"],
    "swiftui":      ["swift", "ios"],
    "jetpack compose": ["kotlin", "android"],

    # Tailwind / Bootstrap imply CSS
    "tailwind":     ["css"],
    "bootstrap":    ["css"],
    "sass":         ["css"],
    "less":         ["css"],

    # State management implies framework
    "redux":        ["react", "javascript"],
    "zustand":      ["react", "javascript"],

    # CI/CD tools imply CI/CD knowledge
    "github actions": ["ci/cd", "git"],
    "gitlab ci":    ["ci/cd", "git"],
    "jenkins":      ["ci/cd"],
    "circleci":     ["ci/cd"],
    "travis ci":    ["ci/cd"],

    # SQL variants imply SQL
    "mysql":        ["sql"],
    "postgresql":   ["sql"],
    "sqlite":       ["sql"],
    "oracle":       ["sql"],
    "sql server":   ["sql"],
    "mariadb":      ["sql"],
    "plsql":        ["sql"],
    "snowflake":    ["sql"],
    "cockroachdb":  ["sql"],
}


def _expand_implied_skills(skills: set[str]) -> set[str]:
    """Expand a skill set with all transitively implied skills.

    e.g. {"next.js"} → {"next.js", "react", "javascript", "html", "css"}
    Uses iterative expansion to resolve chains like:
        next.js → react → javascript + html + css
    """
    expanded = set(skills)
    changed = True
    while changed:
        changed = False
        for skill in list(expanded):
            for implied in SKILL_IMPLIES.get(skill, []):
                if implied not in expanded and implied in ALL_KNOWN_SKILLS:
                    expanded.add(implied)
                    changed = True
    return expanded


# ---------------------------------------------------------------------------
# Regex patterns — built once, cached for performance
# ---------------------------------------------------------------------------

# Skills that need special regex handling (not plain \b word boundaries)
_SPECIAL_PATTERNS: dict[str, re.Pattern] = {}
_STANDARD_PATTERNS: dict[str, re.Pattern] = {}


def _has_nonword_chars(s: str) -> bool:
    """Check if skill contains non-word chars (+, #, .) that break \\b."""
    return bool(re.search(r"[+#.]", s))


def _build_patterns() -> None:
    """Pre-compile regex patterns for all skills with correct boundary logic.

    \\b (word boundary) only works between \\w and \\W characters.
    Skills like 'c++', 'c#', '.net', 'node.js' contain non-word chars
    that break \\b, so they need custom boundary patterns.
    """
    # Boundary that works universally: whitespace, punctuation, or string edge
    # (but NOT +, #, . which are part of skill names)
    _LEFT_B = r"(?:^|(?<=[\s,;:(|/]))"
    _RIGHT_B = r"(?=[\s,;:)|/]|$)"

    for skill in ALL_KNOWN_SKILLS:
        if skill.startswith("."):
            # ".net" — must be preceded by whitespace/start
            _SPECIAL_PATTERNS[skill] = re.compile(
                rf"(?:^|(?<=\s)){re.escape(skill)}{_RIGHT_B}"
            )
        elif len(skill) <= 2 and skill.isalpha():
            # Short alpha skills (c, r, go) — strict standalone matching
            _SPECIAL_PATTERNS[skill] = re.compile(
                rf"{_LEFT_B}{re.escape(skill)}{_RIGHT_B}"
            )
        elif _has_nonword_chars(skill):
            # Skills with +, #, . (c++, c#, node.js, ci/cd, etc.)
            # Can't use \b — use explicit boundary lookarounds
            _SPECIAL_PATTERNS[skill] = re.compile(
                rf"(?:^|(?<=[\s,;:(|/])){re.escape(skill)}{_RIGHT_B}"
            )
        else:
            # Normal alpha-numeric skills — standard \b works fine
            _STANDARD_PATTERNS[skill] = re.compile(
                rf"\b{re.escape(skill)}\b"
            )


_build_patterns()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip noise chars (keep +#. for C++, C#, .NET)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s,;:()|-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Extract & match
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> set[str]:
    """Extract all recognized skills from free text, then expand implications."""
    normalized = _normalize(text)
    directly_found: set[str] = set()

    for skill, pattern in _STANDARD_PATTERNS.items():
        if pattern.search(normalized):
            directly_found.add(skill)

    for skill, pattern in _SPECIAL_PATTERNS.items():
        if pattern.search(normalized):
            directly_found.add(skill)

    # Expand with implied skills
    return _expand_implied_skills(directly_found)


# ---------------------------------------------------------------------------
# "or" / "/" alternative detection
# ---------------------------------------------------------------------------

def _extract_alternative_groups(jd_text: str) -> list[set[str]]:
    """Detect 'X or Y' and 'X/Y' patterns in JD text.

    Strategy: find all known skills in the JD with their text positions,
    then check if any pair of adjacent skills is connected by ' or ' or '/'.
    Groups connected skills as alternatives.

    'Java or Python'       → [{'java', 'python'}]
    'AWS or Azure or GCP'  → [{'aws', 'azure', 'gcp'}]
    'React/Angular/Vue'    → [{'react', 'angular', 'vue'}]
    'ci/cd'                → (skipped, it's a single skill)
    """
    normalized = _normalize(jd_text)

    # Step 1: Find all skill occurrences with positions
    skill_hits: list[tuple[str, int, int]] = []  # (skill_name, start, end)

    for skill, pattern in _STANDARD_PATTERNS.items():
        for m in pattern.finditer(normalized):
            skill_hits.append((skill, m.start(), m.end()))

    for skill, pattern in _SPECIAL_PATTERNS.items():
        for m in pattern.finditer(normalized):
            skill_hits.append((skill, m.start(), m.end()))

    # Sort by position in text
    skill_hits.sort(key=lambda x: x[1])

    # Step 2: Check text between consecutive skills for ' or ' / '/'
    groups: list[set[str]] = []
    seen: set[str] = set()
    current_group: set[str] = set()

    for i in range(len(skill_hits)):
        skill_a, _, end_a = skill_hits[i]

        if i + 1 < len(skill_hits):
            skill_b, start_b, _ = skill_hits[i + 1]
            between = normalized[end_a:start_b].strip()

            # Check if the text between two skills is " or " or "/"
            is_or = between in ("or", "/")
            # Also match patterns like ", or" or " ,or"
            is_or = is_or or re.fullmatch(r",?\s*or\s*,?", between) is not None

            if is_or and skill_a != skill_b:
                if not current_group:
                    current_group.add(skill_a)
                current_group.add(skill_b)
            else:
                # Chain broken — flush current group if valid
                if len(current_group) >= 2 and not current_group & seen:
                    groups.append(current_group)
                    seen.update(current_group)
                current_group = set()
        else:
            # Last skill — flush
            if len(current_group) >= 2 and not current_group & seen:
                groups.append(current_group)
                seen.update(current_group)

    # Handle slash-separated compound terms like "React/Angular/Vue"
    slash_re = re.compile(r"\b([a-z][a-z0-9+#.]*(?:/[a-z][a-z0-9+#.]*)+)\b")
    for match in slash_re.finditer(normalized):
        full_term = match.group(0)
        # Skip if the whole slash-term is a known single skill (e.g. ci/cd)
        if full_term in ALL_KNOWN_SKILLS or full_term in _SPECIAL_PATTERNS:
            continue
        parts = full_term.split("/")
        skills_in_group = {
            p for p in parts
            if p in ALL_KNOWN_SKILLS or p in _SPECIAL_PATTERNS
        }
        if len(skills_in_group) >= 2 and not skills_in_group & seen:
            groups.append(skills_in_group)
            seen.update(skills_in_group)

    return groups


def match_skills(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
) -> dict[str, Any]:
    """Compare resume skills against JD + role profile skills.

    Handles 'X or Y' patterns: when the JD says 'Java or Python',
    matching either one counts as a full match for that requirement.
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    # Detect alternative groups BEFORE flattening
    alt_groups = _extract_alternative_groups(jd_text)
    alt_skill_set = set()  # all skills that belong to an alt group
    for group in alt_groups:
        alt_skill_set.update(group)

    # Build required skill set from JD + role profile
    # Exclude alt-group skills from the flat required set — the JD's
    # "X or Y" intent overrides the role profile's flat listing.
    required_flat = set(jd_skills) - alt_skill_set
    role_key = (role or "").lower().strip()
    if role_key in ROLE_PROFILES:
        for s in ROLE_PROFILES[role_key]["skills"]:
            normalized = s.lower().strip()
            if normalized not in alt_skill_set:
                required_flat.add(normalized)

    # --- Score flat (non-alternative) skills ---
    matched_flat = resume_skills & required_flat
    missing_flat = required_flat - resume_skills

    # --- Score alternative groups ---
    # Each group counts as 1 requirement. Match if candidate has ANY from the group.
    matched_alt_skills: set[str] = set()
    missing_alt_labels: list[str] = []
    alt_matched_count = 0

    for group in alt_groups:
        candidate_has = resume_skills & group
        if candidate_has:
            alt_matched_count += 1
            matched_alt_skills.update(candidate_has)
        else:
            # Show what's missing as "X or Y"
            missing_alt_labels.append(" or ".join(sorted(group)))

    # --- Combine ---
    total_requirements = len(required_flat) + len(alt_groups)
    total_matched = len(matched_flat) + alt_matched_count

    all_matched = sorted(matched_flat | matched_alt_skills)
    all_missing = sorted(missing_flat) + sorted(missing_alt_labels)
    extra = sorted(resume_skills - required_flat - alt_skill_set)

    total = max(total_requirements, 1)
    score = round((total_matched / total) * 100, 2)

    return {
        "matched": all_matched,
        "missing": all_missing,
        "extra": extra,
        "score": score,
        "total_required": total_requirements,
        "matched_count": total_matched,
    }


def get_available_roles() -> list[str]:
    """Return list of supported role names."""
    return sorted(ROLE_PROFILES.keys())


def get_role_weights(role: str | None = None) -> dict[str, float]:
    """Return unified scoring weights (JD-first approach).

    Previously returned role-specific weights, but the JD-first architecture
    means we always use unified weights. The JD drives scoring through skill
    extraction, not through predefined role profiles.

    Role parameter is kept for API compatibility but is advisory only.
    """
    from app.config import UNIFIED_WEIGHTS
    return UNIFIED_WEIGHTS.copy()
