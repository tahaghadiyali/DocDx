# 🩺 DocDx

DocDx is your one stop solution for all the medical related advices and suggestions for who is best Doctor for you to consult in your locality.

> *You describe how you feel. We find who can help.*

**DocDx** (formerly RemedyRadar) is an agentic health specialist recommendation system. Given a natural-language description of symptoms, a location, and optional preferences, it returns a ranked list of suitable specialists nearby — with transparent reasoning for every recommendation.

## ✨ Key Features

- **6-Agent Pipeline** — Intake → Emergency Detection → Specialty Classification → Geo Search → Ranking → Explanation
- **Dual RAG Stores** — Medical knowledge (symptoms→specialties) + Doctor profiles (bios, reviews)
- **Transparent Ranking** — Deterministic scoring with full breakdown per factor (no black-box LLM ranking)
- **Emergency Detection** — Two-layer (keyword + LLM) with zero-tolerance for false negatives
- **Multi-Turn Refinement** — "Show only women doctors", "prioritize experience" — session-aware follow-ups
- **100% Free** — Ollama (local LLM), ChromaDB, PostgreSQL+PostGIS, OpenStreetMap, Chainlit

## 🏗 Architecture

```
User ─→ Chainlit Chat UI ─→ LangGraph Orchestrator
                                │
                    ┌───────────┼───────────────┐
                    ▼           ▼               ▼
              Intake Agent  Emergency       Knowledge Agent
              (Ollama LLM) Detector         (RAG Store #1)
                    │       (Keywords+LLM)      │
                    └──────►┌──┘                │
                            │ (if safe)         │
                            ▼                   ▼
                    Retrieval Agent ──► Ranking Agent ──► Explanation Agent
                    (PostGIS Geo)      (Pure Python)     (RAG Store #2 + LLM)
```

## 🛠 Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| LLM | Ollama + Llama 3.1 8B | Free |
| Embeddings | nomic-embed-text (Ollama) | Free |
| Orchestration | LangGraph | Free |
| Vector DB | ChromaDB | Free |
| Structured DB | PostgreSQL + PostGIS | Free |
| Geo Data | OpenStreetMap + Nominatim | Free |
| Medical Data | MedlinePlus + Curated | Free |
| Frontend | Chainlit | Free |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for PostgreSQL)
- [Ollama](https://ollama.com) installed

### Setup

```bash
# 1. Clone and enter the project
cd remedy-radar

# 2. Install Ollama models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 3. Start PostgreSQL+PostGIS
docker compose up -d

# 4. Install Python dependencies
pip install -e .

# 5. Copy env file
cp .env.example .env

# 6. Generate synthetic data
python scripts/generate_synthetic.py

# 7. Seed the database
python scripts/seed_database.py

# 8. Build vector stores
python scripts/build_vector_stores.py

# 9. Launch RemedyRadar
chainlit run app/chainlit_app.py -w
```

Open http://localhost:8000 and start describing your symptoms!

## 📊 Evaluation

```bash
python -m app.eval.evaluate_specialty     # Specialty mapping precision
python -m app.eval.evaluate_rag           # Groundedness check
python -m app.eval.evaluate_latency       # End-to-end latency
```

## ⚠️ Disclaimer

RemedyRadar is a **portfolio/demo project** and does NOT provide medical diagnosis or treatment advice. It maps symptoms to likely specialties using public medical references. Always consult a qualified healthcare professional for medical decisions.

## 📄 License

MIT
