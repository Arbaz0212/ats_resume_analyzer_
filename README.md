# 📄 ATS Resume Analyzer

An enterprise-grade Applicant Tracking System (ATS) resume analyzer built with **FastAPI**, **HTMX**, **Alpine.js**, **Tailwind CSS**, and **Chart.js**. Features deep LLM integration for semantic analysis, career trajectory evaluation, and AI-powered resume improvement coaching.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![HTMX](https://img.shields.io/badge/HTMX-2.0-purple)
![Ollama](https://img.shields.io/badge/Ollama-AI--Powered-orange)

---

## ✨ Features

### 📊 Batch Analyzer (`/analyze`)
- **Multi-Resume Upload** — Drag & drop up to 50 PDF/DOCX resumes at once with upload progress bars
- **JD-First Scoring** — Job description drives all scoring; role selection is optional/advisory
- **JD File Upload** — Upload JD as PDF/DOCX alongside text paste option
- **Smart Skill Matching** — 150+ skills across 14 categories matched against JD
- **12 Job Role Profiles** — Software Engineer, Data Scientist, DevOps, QA, Product Manager, and more
- **5-Dimension Scoring** — Skills (35%), Experience (25%), Similarity (20%), Projects (12%), Education (8%)
- **Score Normalization** — Prevents dimension bias across scoring dimensions
- **Radar Charts** — Visual score breakdown per candidate (Chart.js)
- **Ranked Results** — Candidates sorted by ATS score with color-coded hiring decisions
- **Qualified Tab** — Filter to only candidates meeting qualification threshold
- **Comparison Matrix** — Side-by-side candidate comparison for 2+ resumes
- **CSV Export** — Download results for offline review

### 🔍 Individual Resume Review (`/review`)
- **Single Resume Deep Dive** — Upload one resume for personalized feedback
- **Optional JD Targeting** — Score against a specific JD or get general quality feedback
- **Score Ring Visualization** — Animated circular score indicator
- **Dimension Breakdown** — Visual bars for skills, experience, projects, education, and similarity
- **AI Action Plan** — 5 prioritized improvement suggestions with before/after examples
- **Career Trajectory Analysis** — Progression patterns, tenure analysis, growth potential
- **Personalized Insights** — AI-generated strengths, gaps, and interview preparation

### 🧠 AI Features (Ollama-Powered)

All AI features are optional and gracefully degrade when Ollama is not running.

| Feature | Description |
|---------|-------------|
| **Semantic Similarity** | Embeddings via `nomic-embed-text` for deep JD matching |
| **Semantic Skill Matching** | LLM identifies skill equivalences (e.g., "REST APIs" ≈ "API Development") |
| **LLM Holistic Scoring** | Multi-dimensional AI scoring blended with deterministic scores |
| **Per-Candidate Insights** | Strengths, gaps, and targeted interview questions |
| **Resume Improvement Suggestions** | 5 prioritized, actionable recommendations with before/after examples |
| **Career Trajectory Analysis** | Progression type, tenure patterns, green/red flags |
| **Achievement Impact Analysis** | Evaluates quantified achievements and impact statements |
| **Candidate Fit Analysis** | Culture fit, growth potential, team composition signals |
| **Red Flag Detection** | Employment gaps, job hopping, career inconsistencies |
| **Experience Relevance** | Context-aware analysis of past role relevance |
| **Executive Hiring Summary** | Batch-level analysis with 3-5 actionable recommendations |
| **Comparative Ranking** | AI-powered candidate ranking across multiple dimensions |

### 🎨 Design & Accessibility
- **WCAG 2.2 AA Compliant** — Keyboard navigable, proper contrast ratios, screen reader support
- **Walmart Design System** — Blue (`#0053e2`) + Spark (`#ffc220`) branding
- **Dark Mode** — Full dark theme support (press `D` to toggle)
- **Responsive** — Mobile-friendly layout

---

## 🚀 Quick Start

```bash
# 1. Clone & enter project
cd idea

# 2. Create virtual environment
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt --index-url https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple --allow-insecure-host pypi.ci.artifacts.walmart.com

# 4. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8899 --timeout-keep-alive 1800

# 5. Open in browser
open http://localhost:8899
```

---

## 🧠 AI Setup (Optional but Recommended)

To enable AI-powered features, install and run [Ollama](https://ollama.ai):

```bash
# Install Ollama
brew install ollama

# Pull required models
ollama pull qwen2.5:7b        # Chat/analysis model (~4.7GB)
ollama pull nomic-embed-text   # Embeddings for semantic similarity (~274MB)

# Start Ollama server
ollama serve
```

The app automatically detects Ollama and enables AI features. Without Ollama, all deterministic scoring features work normally.

---

## ⚙️ Configuration

All settings are configured via environment variables. Copy `.env.example` to `.env` to customize:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:7b` | LLM model for analysis |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TIMEOUT` | `1200` | LLM request timeout (seconds) |
| `MAX_RESUMES` | `50` | Maximum resumes per batch |
| `MAX_FILE_SIZE_MB` | `10` | Maximum file size per upload |
| `MAX_JD_CHARS` | `50000` | Maximum JD text length |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `FEATURE_JD_FILE_UPLOAD` | `true` | Enable JD file upload (PDF/DOCX) |
| `FEATURE_OPTIONAL_ROLE` | `true` | Make role selection optional |
| `FEATURE_IMPROVEMENT_SUGGESTIONS` | `true` | AI resume improvement suggestions |
| `FEATURE_SCORE_NORMALIZATION` | `true` | Dimension score normalization |
| `ENABLE_SEMANTIC_SKILLS` | `true` | Semantic skill matching via LLM |
| `ENABLE_LLM_EXPERIENCE` | `true` | LLM-powered experience analysis |
| `ENABLE_ACHIEVEMENT_ANALYSIS` | `true` | Achievement impact scoring |
| `ENABLE_CAREER_TRAJECTORY` | `true` | Career progression analysis |
| `ENABLE_MULTI_DIM_FIT` | `true` | Multi-dimensional fit analysis |
| `ENABLE_RED_FLAG_DETECTION` | `true` | Red flag detection |
| `ENABLE_COMPARATIVE_RANKING` | `true` | AI comparative ranking |
| `SEMANTIC_SKILL_THRESHOLD` | `0.75` | Minimum similarity for semantic skill match |

---

## 📁 Project Structure

```
idea/
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── app/
    ├── main.py                   # FastAPI app, routes, middleware
    ├── config.py                 # Environment config + feature flags
    ├── services/
    │   ├── pdf_parser.py         # PDF/DOCX text extraction (pdfplumber + python-docx)
    │   ├── scoring_engine.py     # Central ATS scoring orchestrator
    │   ├── skill_matcher.py      # Skill extraction & JD-aware matching (150+ skills)
    │   ├── similarity.py         # TF-IDF + semantic similarity (Ollama embeddings)
    │   ├── section_scorer.py     # Experience/education/project scoring + normalization
    │   ├── llm_service.py        # Ollama LLM integration (chat, insights, suggestions)
    │   ├── semantic_skill_matcher.py  # LLM-powered semantic skill equivalence
    │   ├── experience_analyzer.py     # Context-aware experience relevance analysis
    │   ├── achievement_analyzer.py    # Achievement impact scoring
    │   ├── fit_analyzer.py            # Multi-dimensional candidate fit analysis
    │   └── red_flag_detector.py       # Employment red flag detection
    ├── data/
    │   ├── skill_db.json         # 150+ skills across 14 categories
    │   ├── role_profiles.json    # 12 role profiles with skills
    │   └── jd_templates.json     # 12 pre-built JD templates
    ├── templates/
    │   ├── base.html             # Base layout (Tailwind, Alpine.js, Chart.js CDN)
    │   ├── dashboard.html        # Landing page
    │   ├── analyze.html          # Batch analyzer page
    │   ├── review.html           # Individual resume review page
    │   └── partials/
    │       ├── results_content.html      # Batch results (rankings, charts, tabs)
    │       ├── review_results.html       # Individual review results
    │       ├── ai_candidate_insight.html # AI insight card per candidate
    │       ├── ai_executive_summary.html # Executive hiring summary
    │       ├── comparison_matrix.html    # Side-by-side candidate comparison
    │       ├── jd_parsed_preview.html    # JD parsing preview
    │       ├── loading.html              # Loading animation
    │       └── error_banner.html         # Error display
    └── static/
        ├── css/app.css           # Design system + component styles
        └── js/app.js             # Score rings, theme toggle, keyboard shortcuts
```

---

## 🏗️ Scoring Architecture

```
JD Text ─────────────────────────────────────────────────────────────┐
  │                                                                  │
  ├─→ Skill Extraction (from JD) ────────→ Required Skills           │
  │                                              │                   │
  │   Resume Text                                │                   │
  │        │                                     │                   │
  │        ├─→ Skill Matching ←──────────────────┘                   │
  │        │       └─→ skills_score (35%)                            │
  │        │                                                         │
  │        ├─→ Semantic/TF-IDF Similarity ───→ similarity_score (20%)│
  │        │                                                         │
  │        ├─→ Experience Scoring ───────────→ experience_score (25%)│
  │        │       └─→ Years + Seniority (inferred if missing)       │
  │        │                                                         │
  │        ├─→ Education Scoring ────────────→ education_score (8%)  │
  │        │                                                         │
  │        └─→ Projects Scoring ─────────────→ projects_score (12%) │
  │                                                                  │
  └─→ Score Normalization ───────────→ Normalized Scores (0-100)     │
          │                                                          │
          └─→ Weighted Average ──────→ Base ATS Score                │
                    │                                                │
                    │    ┌─── AI Enhancement (when Ollama running) ──┘
                    │    │
                    │    ├─→ Achievement Impact Bonus (+0-5 pts)
                    │    ├─→ Candidate Fit Modifier (+/-3 pts)
                    │    ├─→ Red Flag Penalty (-0-8 pts)
                    │    └─→ LLM Holistic Score (40% blend)
                    │
                    └─→ Final Score (0-100)
                            │
                            └─→ Hiring Decision:
                                 ≥80% Strong Match (green)
                                 ≥65% Potential Match (blue)
                                 ≥50% Needs Review (amber)
                                 <50% Weak Match (red)
```

---

## 🎨 Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| Frontend | HTMX + Tailwind CSS + Alpine.js |
| Charts | Chart.js (radar, bar, horizontal bar) |
| PDF Parsing | pdfplumber + python-docx |
| NLP | scikit-learn (TF-IDF vectorization) |
| AI/LLM | Ollama (local, no API keys, no data leaves your machine) |
| Data Validation | Pydantic v2 (structured LLM output) |
| HTTP Client | httpx (async Ollama communication) |
| Design System | Walmart Brand (Blue #0053e2, Spark #ffc220) |

---

## 📝 Pages

| Route | Purpose |
|-------|---------|
| `/` | Dashboard — quick start, feature overview, AI status |
| `/analyze` | Batch Analyzer — multi-resume scoring against a JD |
| `/review` | Resume Review — single resume deep-dive with AI coaching |

---

## 🔒 Security

- Security headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy)
- CORS restricted to localhost origins
- File type validation via magic bytes (not just extension)
- Filename sanitization on upload
- No data persistence — all analysis is ephemeral
- Ollama runs 100% locally — no data leaves your machine