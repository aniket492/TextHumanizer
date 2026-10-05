"""Owns the rewrite model lifecycle (lazy/background loading, locking, timeouts)."""
from __future__ import annotations

import threading
import time

from config import Settings
from core.logging import get_logger
from ml.evaluation.metrics import SemanticSimilarity, get_similarity
from ml.inference.baseline import BaselineRewriter
from ml.inference.predictor import ModelTimeoutError, Predictor

logger = get_logger("model_service")


class ModelUnavailableError(RuntimeError):
    pass


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.model_backend
        self.model_id = settings.model_path or settings.model_name
        self.status = "disabled" if self.backend == "baseline" else "not_loaded"
        self.error: str | None = None
        self.baseline = BaselineRewriter()
        self._predictor: Predictor | None = None
        self._ready = threading.Event()
        self._state_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._similarity: SemanticSimilarity | None = None

    # ------------------------------------------------------------------ loading
    @property
    def similarity(self) -> SemanticSimilarity:
        if self._similarity is None:
            self._similarity = get_similarity(self.settings.similarity_model)
        return self._similarity

    def start_loading(self) -> None:
        """Load the model in a background thread (idempotent)."""
        with self._state_lock:
            if self.status != "not_loaded":
                return
            self.status = "loading"
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            logger.info("Loading rewrite model %s ...", self.model_id)
            self._predictor = Predictor(
                self.model_id, device=self.settings.model_device, max_new_tokens=self.settings.model_max_new_tokens
            )
            self.status = "ready"
        except Exception as exc:  # network errors, missing weights, OOM, ...
            logger.exception("Rewrite model failed to load")
            self.error = f"{type(exc).__name__}: {exc}"[:300]
            self.status = "failed"
        finally:
            self._ready.set()

    def warm_up(self) -> None:
        """Preload similarity model and (optionally) the rewrite model."""
        try:
            _ = self.similarity.method
        except Exception:  # pragma: no cover - logged by metrics
            pass
        if self.backend != "baseline":
            self.start_loading()

    def predictor(self, wait_seconds: float = 0.0) -> Predictor | None:
        """Return the predictor if usable. In 'transformers' mode a missing model is an error."""
        if self.backend == "baseline":
            return None
        if self.status == "not_loaded":
            self.start_loading()
        if self.status == "loading" and wait_seconds > 0:
            self._ready.wait(wait_seconds)
        if self.status == "ready":
            return self._predictor
        if self.backend == "transformers":
            detail = self.error or f"model status is '{self.status}'"
            raise ModelUnavailableError(f"The rewrite model is not available ({detail}).")
        return None

    # ---------------------------------------------------------------- inference
    def generate(self, text: str, style: str, intensity: str, max_time: float) -> list[str]:
        predictor = self._predictor
        if predictor is None:
            raise ModelUnavailableError("The rewrite model is not loaded.")
        start = time.perf_counter()
        if not self._gen_lock.acquire(timeout=max(max_time, 0.1)):
            raise ModelTimeoutError("Timed out waiting for the model (server busy).")
        try:
            remaining = max_time - (time.perf_counter() - start)
            if remaining <= 1.0:
                raise ModelTimeoutError("No time left for generation.")
            return predictor.generate(
                text,
                style=style,
                intensity=intensity,
                num_candidates=self.settings.model_num_candidates,
                max_time=remaining,
            )
        finally:
            self._gen_lock.release()

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model_id if self.backend != "baseline" else None,
            "status": self.status,
            "device": self._predictor.loaded.device if self._predictor else None,
            "error": self.error,
            "baseline": self.baseline.name,
            "similarity": self._similarity.method if self._similarity else "not_loaded",
        }


_service: ModelService | None = None
_service_lock = threading.Lock()


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                from config import settings

                _service = ModelService(settings)
    return _service
