"""Application configuration via environment variables."""

import os

# --- File limits ---
MAX_RESUMES: int = int(os.getenv("MAX_RESUMES", "50"))
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_JD_CHARS: int = int(os.getenv("MAX_JD_CHARS", "50000"))
MIN_JD_CHARS: int = int(os.getenv("MIN_JD_CHARS", "50"))

# --- Ollama (local LLM) ---
# Ollama serves an OpenAI-compatible API on localhost.
# No API key needed — everything runs locally on your machine.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "1200"))

# --- Unified Scoring Weights ---
# JD-first approach: same weights regardless of role selection.
# Role is optional/advisory; JD drives scoring through skill extraction.
UNIFIED_WEIGHTS: dict[str, float] = {
    "skills": 0.35,       # Technical skill match from JD
    "experience": 0.25,   # Years & seniority alignment
    "similarity": 0.20,   # Semantic similarity to JD
    "projects": 0.12,     # Relevant project experience
    "education": 0.08,    # Educational background match
}

# --- Feature Flags ---
# Enable/disable features for safe rollback during deployment
FEATURE_JD_FILE_UPLOAD: bool = os.getenv("FEATURE_JD_FILE_UPLOAD", "true").lower() == "true"
FEATURE_OPTIONAL_ROLE: bool = os.getenv("FEATURE_OPTIONAL_ROLE", "true").lower() == "true"
FEATURE_IMPROVEMENT_SUGGESTIONS: bool = os.getenv("FEATURE_IMPROVEMENT_SUGGESTIONS", "true").lower() == "true"
FEATURE_SCORE_NORMALIZATION: bool = os.getenv("FEATURE_SCORE_NORMALIZATION", "true").lower() == "true"

# --- LLM-Enhanced Scoring Feature Flags ---
# Phase 3: Deep LLM integration for semantic analysis
ENABLE_SEMANTIC_SKILLS: bool = os.getenv("ENABLE_SEMANTIC_SKILLS", "true").lower() == "true"
ENABLE_LLM_EXPERIENCE: bool = os.getenv("ENABLE_LLM_EXPERIENCE", "true").lower() == "true"
ENABLE_ACHIEVEMENT_ANALYSIS: bool = os.getenv("ENABLE_ACHIEVEMENT_ANALYSIS", "true").lower() == "true"
ENABLE_CAREER_TRAJECTORY: bool = os.getenv("ENABLE_CAREER_TRAJECTORY", "true").lower() == "true"
ENABLE_MULTI_DIM_FIT: bool = os.getenv("ENABLE_MULTI_DIM_FIT", "true").lower() == "true"
ENABLE_RED_FLAG_DETECTION: bool = os.getenv("ENABLE_RED_FLAG_DETECTION", "true").lower() == "true"
ENABLE_COMPARATIVE_RANKING: bool = os.getenv("ENABLE_COMPARATIVE_RANKING", "true").lower() == "true"

# Semantic matching configuration
SEMANTIC_SKILL_THRESHOLD: float = float(os.getenv("SEMANTIC_SKILL_THRESHOLD", "0.75"))

# LLM timeout configuration per analysis type (seconds)
LLM_ANALYSIS_TIMEOUT: int = int(os.getenv("LLM_ANALYSIS_TIMEOUT", "1200"))
