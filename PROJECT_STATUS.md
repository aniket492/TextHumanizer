# PROJECT_STATUS

**Current phase:** 14 (final integration test)

**Completed:**
- Phases 1-3: FastAPI app, config, PostgreSQL (docker, host port 5433), SQLAlchemy models, Alembic `0001_initial`, JWT auth (bcrypt + PyJWT)
- Phase 4: dataset download (pinned revision) / inspect / clean / splits / SFT formatting -> 2611 / 290 / 294
- Phases 5-6: text analysis + quality metrics (`ml/evaluation/metrics.py`), rule baseline (`ml/inference/baseline.py`)
- Phase 7: Qwen2.5-0.5B-Instruct inference (CPU), LoRA fine-tune path (smoke run OK, adapter loads)
- Phase 8: `ml/evaluation/evaluate.py` -> `reports/eval_report*.md|json`
- Phases 9-11: API routes, React frontend (dashboard, history, auth), usage tracking
- Phase 12: pytest 54 passed; vitest 5 passed; `npm run build` OK
- Phase 13: Dockerfiles, nginx, compose

**Current task:** Docker build + end-to-end smoke test.

**Known issues / constraints:**
- CPU-only machine: LLM ~9-15 s per ~100-word chunk; long texts finish with the rule engine after the 90 s budget (reported in `notes`).
- Full LoRA fine-tune needs a GPU (~46 s/step on CPU).
- Grammar score is heuristic (no LanguageTool/Java dependency).

**Next task:** see README "Roadmap".
