"""Hybrid text similarity: semantic embeddings (Ollama) + lexical fallback.

When Ollama is running with an embedding model:
  1. nomic-embed-text cosine similarity (semantic understanding)
  2. Jaccard keyword overlap (exact term matching)
  Blend: 70% semantic + 30% Jaccard

When Ollama is unavailable:
  1. TF-IDF cosine similarity (structural overlap)
  2. Jaccard keyword overlap
  Blend: 60% TF-IDF + 40% Jaccard
"""

from __future__ import annotations

import logging
import re

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic similarity (Ollama embeddings)
# ---------------------------------------------------------------------------

def _get_embedding(text: str) -> list[float] | None:
    """Get embedding vector from Ollama's /api/embed endpoint."""
    # Ollama's native embedding endpoint (not the OpenAI-compat one)
    base = OLLAMA_BASE_URL.replace("/v1", "")
    try:
        resp = httpx.post(
            f"{base}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text[:8000]},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None
    except Exception:
        logger.debug("Ollama embedding failed, will fall back to TF-IDF")
        return None


def _semantic_similarity(text_a: str, text_b: str) -> float | None:
    """Compute cosine similarity using Ollama embeddings.

    Returns 0.0–1.0, or None if embeddings unavailable.
    """
    vec_a = _get_embedding(text_a)
    if vec_a is None:
        return None
    vec_b = _get_embedding(text_b)
    if vec_b is None:
        return None

    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# Lexical similarity (TF-IDF fallback)
# ---------------------------------------------------------------------------

def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity (0.0–1.0)."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        sublinear_tf=True,
    )
    vectors = vectorizer.fit_transform([text_a, text_b])
    return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])


# ---------------------------------------------------------------------------
# Jaccard overlap
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was",
    "will", "from", "have", "has", "been", "our", "your", "you",
    "not", "but", "all", "can", "had", "her", "one", "who",
    "their", "there", "what", "about", "which", "when", "make",
    "like", "than", "each", "other", "into", "more", "some",
}


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard index over meaningful tokens (0.0–1.0)."""
    def tokenize(t: str) -> set[str]:
        return set(re.findall(r"\b[a-z][a-z0-9+#.]{1,}\b", t.lower())) - _STOPWORDS

    a, b = tokenize(text_a), tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_similarity(
    resume_text: str,
    jd_text: str,
) -> tuple[float, str]:
    """Return (score 0–100, method_used).

    Uses Ollama semantic embeddings when available, TF-IDF as fallback.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0, "none"

    jaccard = _jaccard_similarity(resume_text, jd_text)

    # Try semantic first (Ollama embeddings)
    semantic = _semantic_similarity(resume_text, jd_text)
    if semantic is not None:
        blended = (semantic * 0.7) + (jaccard * 0.3)
        return round(blended * 100, 2), "semantic"

    # Fall back to TF-IDF
    tfidf = _tfidf_similarity(resume_text, jd_text)
    blended = (tfidf * 0.6) + (jaccard * 0.4)
    return round(blended * 100, 2), "tfidf"
