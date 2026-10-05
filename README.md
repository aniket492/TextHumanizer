# AI Humanizer

A web application that rewrites stiff, mechanical or AI-assisted text into natural, readable writing
**while preserving meaning, facts, names, numbers, dates, citations, URLs and technical terms**.
Every rewrite is checked with transparent, measurable quality metrics.

> This project does **not** claim or attempt to guarantee any AI-detector result, and it contains no
> detector-evasion tricks. It focuses on genuine rewriting and measurable writing quality.

---

## Contents

1. [Architecture](#architecture)
2. [Requirements](#requirements)
3. [Quick start with Docker](#quick-start-with-docker)
4. [Local development](#local-development)
5. [Dataset pipeline](#dataset-pipeline)
6. [Models](#models)
7. [Training (optional LoRA fine-tune)](#training-optional-lora-fine-tune)
8. [Inference](#inference)
9. [Evaluation](#evaluation)
10. [API](#api)
11. [Testing](#testing)
12. [Environment variables](#environment-variables)
13. [Deployment](#deployment)
14. [Troubleshooting](#troubleshooting)
15. [Data and model sources](#data-and-model-sources)

---

## Architecture

```
React (Vite) ──HTTP/JSON──> FastAPI ──> services ──> ml/ (pure Python, no web code)
   frontend/                backend/      │             ├─ preprocessing/  cleaning, normalisation
                              │           │             ├─ inference/      rule baseline, LLM loader/predictor
                              ▼           │             ├─ evaluation/     quality metrics, reports
                         PostgreSQL       │             └─ training/       LoRA fine-tuning
                    (users, history,      └─ model_service: lazy/background model loading, locking, timeouts
                     usage; Alembic)
```

**Humanization pipeline** (`backend/services/humanizer_service.py`):

```
input → validation → segmentation (paragraph-aligned, ≤110-word chunks of whole sentences)
      → analysis (repeated phrases) → candidates per chunk:
            • LLM rewrite (if loaded)          • LLM rewrite + rule cleanup
            • rule-based rewrite (always)
      → quality gates per candidate:
            semantic similarity ≥ threshold (by intensity) · URLs/e-mails/citations/numbers/dates kept verbatim
            · numbers & negations consistent · length ratio within style bounds · grammar heuristic
      → best passing candidate (if none passes, the original wording is kept and the user is told)
      → assembly → final quality report
```

Key design points:

- **ML is isolated from the API.** `backend/app.py` only wires routes, middleware and error handlers.
- **Graceful degradation.** `MODEL_BACKEND=auto` loads the LLM in the background; until it is ready
  (or if it fails/times out) the rule engine serves requests and the response says so in `notes`.
- **No silent fact changes.** Candidates that drop a number, date, URL or citation are rejected.
- **Honest reporting.** Each response includes `engine`, `fallback_used`, `notes` and a quality report.

Project layout follows the requested structure (`backend/`, `ml/`, `data/`, `models/`, `scripts/`,
`frontend/`, `tests/`, `docker/`). Additions: `backend/alembic/` (migrations), `ml/inference/baseline.py`
(rule engine), `scripts/download_model.py`, `reports/` (generated reports), `PROJECT_STATUS.md`.

## Requirements

| Tool | Version used | Notes |
|---|---|---|
| Python | 3.12 or 3.13 | Docker image uses 3.12; developed on 3.13 |
| Node.js | 20+ (22 used) | for the frontend |
| Docker + Compose | Docker 24+ | runs PostgreSQL locally and the full stack |
| RAM | 8 GB minimum, 16 GB recommended | the 0.5B model uses ~2.5 GB on CPU |
| GPU | optional | needed only for practical fine-tuning |

## Quick start with Docker

```bash
cp .env.example .env
# Fill in POSTGRES_PASSWORD and JWT_SECRET, e.g.:
python -c "import secrets; print(secrets.token_urlsafe(48))"

docker compose up --build
```

- Frontend: http://localhost:8080 (nginx proxies `/api` and `/health` to the backend)
- API docs: http://localhost:8000/docs
- PostgreSQL: localhost:5433

The backend runs `alembic upgrade head` on start, then preloads the models into the `hf-cache` volume
(first start downloads ~1 GB; until then requests are served by the rule engine).
To use a fine-tuned adapter, put it in `./models/final` and set `MODEL_PATH=/app/models/final` in `.env`.

## Local development

### 1. Python environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build (skip for CUDA)
pip install -r backend/requirements.txt                              # includes the spaCy en_core_web_sm model
```

### 2. PostgreSQL

```bash
cp .env.example .env            # set POSTGRES_PASSWORD and JWT_SECRET
docker compose up -d postgres   # exposed on localhost:5433
```

(Any PostgreSQL 14+ works; point `DATABASE_URL` at it.)

### 3. Backend

```bash
cp backend/.env.example backend/.env   # set DATABASE_URL (same password) and JWT_SECRET
cd backend
alembic upgrade head
uvicorn app:app --reload --port 8000
```

Optional: pre-download models so the first request does not wait for them:
`python scripts/download_model.py` (resumable, retries on network errors).

### 4. Frontend

```bash
cd frontend
cp .env.example .env     # VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev              # http://localhost:5173
```

## Dataset pipeline

Run from the project root, in order. Raw files are written once and never modified.

```bash
python scripts/download_dataset.py   # -> data/raw/gohumanize/{train,test}.jsonl + manifest.json (pinned revision)
python scripts/inspect_dataset.py    # structure, columns, samples, missing values, duplicates, lengths -> reports/
python scripts/clean_dataset.py      # -> data/processed/{train,test}_clean.jsonl + cleaning_report.json
python scripts/create_splits.py      # -> data/train, data/validation, data/test (seed 42)
python scripts/prepare_dataset.py    # -> data/*/sft.jsonl (chat format, same prompt as inference)
```

Cleaning (`ml/preprocessing/`): whitespace/invisible-character normalisation only (punctuation and wording
are preserved); removes empty, corrupted (mojibake, markup, replacement characters), too short/long,
identical and duplicate pairs; removes targets that **repeat their own passages** and targets containing
**numbers absent from the input** (they would teach the model to invent facts); checks train/test leakage.

Result of the current run:

| | raw | kept | main drop reasons |
|---|---|---|---|
| train | 2,957 | 2,901 → 2,611 train + 290 validation (stratified by category) | 44 unsupported numbers, 7 internal repetition, 3 duplicate outputs |
| test | 300 | 294 | 5 unsupported numbers, 1 internal repetition |

The official test split is kept as the test set so results stay comparable with the dataset card.

**Additional datasets.** Other open human/AI text corpora were considered, but most are AI-*detection*
datasets (unpaired human vs. machine text) rather than meaning-preserving rewrite pairs, or carry
non-commercial / unclear licences. None was added: they would not improve rewrite quality and would add
legal risk. A known gap remains: the dataset has no casual modern writing (see its card).

## Models

| Purpose | Default | Licence | Why |
|---|---|---|---|
| Rewrite LLM | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | instruction-following, runs on CPU (~2.5 GB RAM), LoRA-friendly |
| Semantic similarity | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | small, fast, strong sentence embeddings |
| NER / POS | spaCy `en_core_web_sm` 3.8 | MIT | entity preservation and grammar heuristics |
| Rule baseline | `ml/inference/baseline.py` | this repo | always available, deterministic, no download |

Everything is configured by environment variables (`MODEL_BACKEND`, `MODEL_NAME`, `MODEL_PATH`,
`MODEL_DEVICE`, `SIMILARITY_MODEL`). Larger options for GPU machines: `Qwen/Qwen2.5-1.5B-Instruct`,
`Qwen/Qwen2.5-3B-Instruct`, or `gohumanize/gohumanize-open-humanizer` (Qwen3-4B full fine-tune on this
dataset, Apache-2.0; ~8 GB in bf16, not practical on CPU). Seq2seq models (e.g. `google/flan-t5-base`)
are also supported by the loader.

**Measured CPU speed** (Intel i5-12450H): model load ~50 s; ~9 s per 55-word chunk; ~50 s per
200-word passage. Requests have a 90 s budget (`MODEL_TIMEOUT_SECONDS`); chunks that do not fit are
handled by the rule engine and the response says so.

## Training (optional LoRA fine-tune)

```bash
# Prove the pipeline end to end on CPU (2 steps, minutes):
python -m ml.training.train --max-steps 2 --max-train-samples 8 --batch-size 1 --grad-accum 2 \
    --max-seq-length 768 --output-dir models/checkpoints/smoke

# Real run (GPU recommended; ~46 s/step on CPU makes a full epoch impractical there):
python -m ml.training.train --epochs 2
```

- LoRA r=16, alpha=32 on all attention/MLP projections; loss on assistant tokens only; cosine schedule.
- Examples longer than `--max-seq-length` are skipped (never truncated mid-target).
- Output: adapter + tokenizer in `models/final/` (tokenizer also in `models/tokenizer/`), checkpoints in
  `models/checkpoints/`, config and metrics in `models/final/training_summary.json`.
- Serve it: `MODEL_PATH=../models/final` (the adapter is merged into its base model at load time).

## Inference

- Through the API: `POST /api/humanize` (see below).
- From the command line:

```bash
python -m ml.evaluation.test_model --engine baseline --text "Furthermore, it is important to note that ..."
python -m ml.evaluation.test_model --engine model --style casual --intensity strong   # uses a test sample
```

Styles: `natural, professional, academic, casual, simple, concise, detailed`.
Intensities: `light, medium, strong` — they control how much the wording changes and the minimum meaning
similarity required; they are not a detector-evasion setting.

## Evaluation

```bash
python -m ml.evaluation.evaluate --engines identity baseline --limit 294   # full test split
python -m ml.evaluation.evaluate --engines model --limit 12 --tag model    # LLM (slow on CPU)
```

Reports: `reports/eval_report*.md` and `.json` (per-record and per-category). Metrics: semantic
similarity, key-item preservation, factual consistency (numbers and negations), grammar heuristic,
readability (Flesch), repetition, stock-phrase rate, length ratio, sentence-length variation, and
similarity to the human-written reference. **No AI-detector scores are used**; if you add one for
research, treat it as an external probabilistic signal, not ground truth.

Results on the test split (natural / medium):

| metric | identity (control) | rule baseline (294 records) |
|---|---|---|
| semantic similarity to input | 1.000 | 0.990 |
| key items fully preserved | 100% | 99.7% |
| factual consistency | 1.000 | 1.000 |
| grammar heuristic | 0.991 | 0.993 |
| stock phrases per 100 words | 0.88 | **0.05** |
| readability (Flesch) | 39.8 | 41.9 |
| similarity to human reference | 0.874 | 0.870 |

The rule engine removes mechanical phrasing reliably and safely, but it does not move the text closer to the
human reference: deeper style transfer needs the model, ideally fine-tuned on a GPU. LLM results are in
`reports/eval_report_model.md`.

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/` | – | service info |
| GET | `/health` | – | DB + model status (503 if the DB is down) |
| POST | `/api/auth/register` | – | `{name, email, password}` → token + user (409 on duplicate e-mail) |
| POST | `/api/auth/login` | – | `{email, password}` → token + user |
| GET | `/api/auth/me` | Bearer | current user |
| POST | `/api/humanize` | Bearer | `{text, tone, intensity, save_history}` → output, engine, notes, quality |
| POST | `/api/analyze` | – | `{text, compare_text?}` → statistics, readability, repetition, similarity |
| GET | `/api/history?page=&page_size=` | Bearer | paginated history |
| GET | `/api/history/{id}` | Bearer | one item (404 for other users' items) |
| DELETE | `/api/history/{id}` | Bearer | delete (204) |
| GET | `/api/usage` | Bearer | request count, words processed, history count |

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/login -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"Your-passw0rd"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s -X POST localhost:8000/api/humanize -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"text":"Furthermore, it is important to note that our team utilizes many tools in order to ship faster.","tone":"natural","intensity":"medium"}'
```

Errors always have the shape `{"error": {"code", "message", "details?"}}` — e.g. `empty_input`,
`input_too_long` (413), `invalid_json` (400), `validation_error` (422), `unauthorized` (401),
`model_unavailable` (503), `model_timeout` (504), `database_unavailable` (503), `internal_error` (500).
Stack traces and secrets are never returned. Logs contain method, path, status, duration and a request
id — never passwords, tokens or user text.

## Testing

```bash
pytest                                   # 54 backend tests (SQLite, rule engine, TF-IDF similarity: fast, no downloads)
TEST_DATABASE_URL=postgresql+psycopg://humanizer:<pw>@localhost:5433/humanizer_test pytest   # against PostgreSQL
cd frontend && npm test && npm run build # 5 frontend tests + production build
```

Covered: root/health, registration, login, token expiry/forgery, protected routes, validation (empty, too
long, invalid JSON, bad enums), all style × intensity combinations, fact preservation, analysis metrics,
history CRUD + pagination + per-user isolation, usage counters, cascade deletes, and Alembic
upgrade/downgrade.

## Environment variables

Backend (`backend/.env`, or container env):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://humanizer:humanizer@localhost:5433/humanizer` | SQLAlchemy URL |
| `JWT_SECRET` | – (required in production) | HMAC secret; dev falls back to an ephemeral secret |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 60 | token lifetime |
| `API_BASE_URL` | `http://localhost:8000` | public API URL (informational) |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8080` | allowed browser origins |
| `MODEL_BACKEND` | `auto` | `auto` / `transformers` (model required) / `baseline` (rules only) |
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model id |
| `MODEL_PATH` | – | local fine-tuned model or LoRA adapter (overrides `MODEL_NAME`) |
| `MODEL_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `MODEL_MAX_NEW_TOKENS` / `MODEL_TIMEOUT_SECONDS` / `MODEL_NUM_CANDIDATES` | 384 / 90 / 1 | generation limits |
| `MODEL_PRELOAD` | `false` (Docker: `true`) | load models at startup |
| `SIMILARITY_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | empty = TF-IDF fallback |
| `MAX_INPUT_WORDS` / `MAX_INPUT_CHARS` | 1500 / 12000 | input limits |

Compose (`.env`): `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`, `JWT_SECRET`,
`BACKEND_PORT`, `FRONTEND_PORT`, `CORS_ORIGINS`, `MODEL_*`. Frontend: `VITE_API_BASE_URL`.

## Deployment

- Build and run with `docker compose up --build -d` on any Docker host; put a TLS-terminating proxy
  (Caddy, Traefik, a cloud load balancer) in front of the `frontend` service.
- Set `ENVIRONMENT=production`, strong `JWT_SECRET` / `POSTGRES_PASSWORD`, and `CORS_ORIGINS` to your domain.
- Use a managed PostgreSQL by pointing `DATABASE_URL` at it; migrations run on backend start.
- For GPU hosts install the CUDA build of PyTorch in the backend image and set `MODEL_DEVICE=cuda`;
  you can then raise `MODEL_NUM_CANDIDATES` and use a larger `MODEL_NAME`.
- The model is single-flight per process (a generation lock); scale out with more backend replicas.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `engine: baseline-rules-v1` and a note that the model is loading | first start downloads the model; wait or run `python scripts/download_model.py` |
| Model download `Read timed out` | rerun `scripts/download_model.py` (resumes; uses `hf_xet` and a 120 s timeout) |
| `/health` returns 503 `database: unavailable` | start PostgreSQL (`docker compose up -d postgres`) and check `DATABASE_URL` |
| `docker compose` says `Set POSTGRES_PASSWORD` | create `.env` from `.env.example` |
| Docker build: `lookup registry-1.docker.io: no such host` | network/DNS issue; check connectivity and retry |
| Humanize takes long / notes mention time limit | CPU inference; shorten text, use a GPU, or `MODEL_BACKEND=baseline` |
| 401 after restarting the backend in dev | `JWT_SECRET` was unset (ephemeral secret); set it in `backend/.env` |
| Windows: symlink warning from Hugging Face | harmless; enable Developer Mode to save disk space |

## Data and model sources

| Resource | Source | Licence |
|---|---|---|
| GoHumanize Open Humanizer Dataset (rev `badba6c`) | https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset | CC-BY 4.0 — cite: GoHumanize team (2026), https://doi.org/10.5281/zenodo.22843083 |
| Qwen2.5-0.5B-Instruct | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct | Apache-2.0 |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 |
| spaCy en_core_web_sm 3.8.0 | https://github.com/explosion/spacy-models | MIT |

The dataset's human text is public domain (pre-1929 Project Gutenberg originals and US federal government
publications); its AI-styled inputs were generated by gpt-4o-mini, Llama-3.3-70B and DeepSeek-chat.

## License

Code: MIT (see `LICENSE`). Data and models keep their own licences listed above.
