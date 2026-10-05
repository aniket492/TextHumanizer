"""Humanization pipeline.

validation -> segmentation -> analysis -> rewrite (model + rule baseline) ->
semantic/entity/grammar checks -> candidate selection -> assembly -> quality report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from config import Settings
from core.constants import Intensity, Style
from core.logging import get_logger
from ml.evaluation.metrics import repeated_phrases
from ml.inference.predictor import ModelTimeoutError
from ml.preprocessing.normalize_text import word_count
from services.model_service import ModelService, ModelUnavailableError
from services.quality_service import Candidate, QualityService
from utils.text_utils import assemble, chunk_text
from utils.validators import validate_text

logger = get_logger("humanizer")

CHUNK_WORDS = 110


@dataclass
class HumanizeResult:
    original_text: str
    output_text: str
    tone: Style
    intensity: Intensity
    engine: str
    fallback_used: bool
    notes: list[str]
    input_word_count: int
    output_word_count: int
    chunks: int
    processing_ms: int
    quality: dict


class HumanizerService:
    def __init__(self, settings: Settings, models: ModelService):
        self.settings = settings
        self.models = models
        self.quality = QualityService(models.similarity)

    def humanize(self, text: str, tone: Style, intensity: Intensity) -> HumanizeResult:
        start = time.perf_counter()
        clean = validate_text(
            text,
            min_words=self.settings.min_input_words,
            max_words=self.settings.max_input_words,
            max_chars=self.settings.max_input_chars,
        )
        chunks = chunk_text(clean, CHUNK_WORDS)
        notes: list[str] = []
        repeated = repeated_phrases(clean, top=3)
        if repeated:
            notes.append("Repeated phrases in the input: " + ", ".join(f"“{r['text']}”" for r in repeated))

        predictor = self.models.predictor(wait_seconds=0)
        if predictor is None and self.models.backend == "auto":
            if self.models.status == "loading":
                notes.append("The language model is still loading; the rule-based engine was used for this request.")
            elif self.models.status == "failed":
                notes.append("The language model is unavailable; the rule-based engine was used.")

        deadline = start + self.settings.model_timeout_seconds
        outputs: list[str] = []
        sources: set[str] = set()
        kept_original = 0
        model_errors = 0
        for chunk in chunks:
            baseline_text, changes = self.models.baseline.rewrite(chunk.text, tone.value, intensity.value)
            candidates = [Candidate(baseline_text, "baseline", changes)]
            if predictor is not None and model_errors < 2:
                remaining = deadline - time.perf_counter()
                try:
                    if remaining <= 2:
                        raise ModelTimeoutError("request deadline reached")
                    for out in self.models.generate(chunk.text, tone.value, intensity.value, max_time=remaining):
                        candidates.insert(0, Candidate(out, "model"))
                        # Small models still emit stock phrasing; also try the output after rule cleanup.
                        polished, polish_changes = self.models.baseline.rewrite(out, tone.value, "medium")
                        if polished != out:
                            candidates.insert(1, Candidate(polished, "model", polish_changes))
                except ModelTimeoutError as exc:
                    model_errors = 2  # stop trying the model for the remaining chunks
                    logger.warning("Model timeout: %s", exc)
                    if self.models.backend == "transformers":
                        raise
                except ModelUnavailableError:
                    raise
                except Exception:
                    model_errors += 1
                    logger.exception("Model generation failed for a chunk")
                    if self.models.backend == "transformers" and model_errors >= 2:
                        raise ModelUnavailableError("The rewrite model failed repeatedly.")

            best, scored = self.quality.select(chunk.text, candidates, tone, intensity)
            if best is None:
                kept_original += 1
                outputs.append(chunk.text)
                sources.add("original")
                logger.info(
                    "No candidate passed quality gates: %s",
                    "; ".join(f"{s.candidate.source}: {', '.join(s.reasons)}" for s in scored),
                )
            else:
                outputs.append(best.candidate.text)
                sources.add(best.candidate.source)

        if model_errors >= 2 and predictor is not None:
            notes.append("The model ran out of time on part of the text; the rule-based engine handled the rest.")
        if kept_original:
            notes.append(
                f"{kept_original} of {len(chunks)} passage(s) kept their original wording because no rewrite "
                "preserved the meaning and key details reliably."
            )

        output = assemble(chunks, outputs)
        if "model" in sources:
            engine = f"model:{self.models.model_id}" + ("+rules" if "baseline" in sources else "")
        elif "baseline" in sources:
            engine = self.models.baseline.name
        else:
            engine = "unchanged"
        fallback_used = predictor is None or "model" not in sources or len(sources) > 1
        quality = self.quality.report(clean, output, intensity)
        if output.strip() == clean.strip():
            notes.append("No safe improvements were found for this text at the selected settings.")

        elapsed = int((time.perf_counter() - start) * 1000)
        logger.info(
            "humanize tone=%s intensity=%s words=%d chunks=%d engine=%s ms=%d",
            tone.value, intensity.value, word_count(clean), len(chunks), engine, elapsed,
        )
        return HumanizeResult(
            original_text=clean,
            output_text=output,
            tone=tone,
            intensity=intensity,
            engine=engine,
            fallback_used=fallback_used,
            notes=notes,
            input_word_count=word_count(clean),
            output_word_count=word_count(output),
            chunks=len(chunks),
            processing_ms=elapsed,
            quality=quality,
        )
