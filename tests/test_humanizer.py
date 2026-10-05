from __future__ import annotations

import pytest

from core.constants import Intensity, Style
from ml.evaluation.metrics import extract_key_items
from ml.inference.baseline import BaselineRewriter
from services.model_service import get_model_service
from services.humanizer_service import HumanizerService
from config import settings
from utils.text_utils import assemble, chunk_text
from utils.validators import TextValidationError, validate_text


@pytest.fixture(scope="module")
def service() -> HumanizerService:
    return HumanizerService(settings, get_model_service())


def test_baseline_removes_stock_phrases_and_keeps_facts(sample_text):
    out, changes = BaselineRewriter().rewrite(sample_text, "natural", "medium")
    lowered = out.lower()
    for phrase in ("it is important to note", "in order to", "plays a crucial role", "utilize", "a wide range of"):
        assert phrase not in lowered
    for must_keep in ("2023", "87%", "(Smith, 2023)", "https://example.org/report", "National Center for Education Statistics"):
        assert must_keep in out
    assert "filler" in changes and "transitions" in changes
    assert out[0].isupper()


def test_baseline_styles_change_register():
    text = "We do not think it is ready. They are not sure that the plan will work for everyone involved here."
    casual, _ = BaselineRewriter().rewrite(text, "casual", "medium")
    assert "don't" in casual and "it's" in casual
    academic, _ = BaselineRewriter().rewrite("We don't think it's ready, but they're not sure it'll work.", "academic", "medium")
    assert "don't" not in academic and "do not" in academic


def test_baseline_protects_quotes_and_is_deterministic():
    text = 'The CEO said "in order to utilize our assets we must act" and it is important to note that sales rose.'
    r = BaselineRewriter()
    out1, _ = r.rewrite(text, "natural", "strong")
    out2, _ = r.rewrite(text, "natural", "strong")
    assert out1 == out2
    assert '"in order to utilize our assets we must act"' in out1


def test_validation_rules():
    with pytest.raises(TextValidationError) as e:
        validate_text("   ", min_words=3, max_words=100, max_chars=1000)
    assert e.value.code == "empty_input"
    with pytest.raises(TextValidationError) as e:
        validate_text("two words", min_words=3, max_words=100, max_chars=1000)
    assert e.value.code == "input_too_short"
    with pytest.raises(TextValidationError) as e:
        validate_text("word " * 200, min_words=3, max_words=100, max_chars=10000)
    assert e.value.code == "input_too_long" and e.value.status_code == 413
    with pytest.raises(TextValidationError) as e:
        validate_text("12345 67890 !!!! #### $$$$ %%%%", min_words=3, max_words=100, max_chars=1000)
    assert e.value.code == "invalid_text"
    assert validate_text("Hello   there friend,  how are you?", min_words=3, max_words=100, max_chars=1000) == \
        "Hello there friend, how are you?"


def test_chunking_preserves_paragraphs():
    text = ("First sentence here. " * 40).strip() + "\n\nSecond paragraph is short."
    chunks = chunk_text(text, max_words=50)
    assert len({c.paragraph for c in chunks}) == 2
    assert all(len(c.text.split()) <= 60 for c in chunks)
    assert assemble(chunks, [c.text for c in chunks]).split("\n\n")[1] == "Second paragraph is short."


def test_pipeline_preserves_key_items(service, sample_text):
    result = service.humanize(sample_text, Style.natural, Intensity.medium)
    assert result.output_text and result.output_text != sample_text
    items = extract_key_items(sample_text, use_ner=False)
    for url in items["urls"]:
        assert url in result.output_text
    for c in items["citations"]:
        assert c in result.output_text
    assert "87%" in result.output_text and "2023" in result.output_text
    q = result.quality
    assert q["passed"] is True
    assert q["factual_consistency"] >= 0.85
    assert 0.5 <= q["length_ratio"] <= 1.5
    assert result.engine.startswith("baseline")


@pytest.mark.parametrize("style", list(Style))
@pytest.mark.parametrize("intensity", list(Intensity))
def test_pipeline_all_styles_and_intensities(service, sample_text, style, intensity):
    result = service.humanize(sample_text, style, intensity)
    assert result.output_text.strip()
    assert result.quality["factual_consistency"] >= 0.85
    assert "https://example.org/report" in result.output_text


def test_pipeline_rejects_empty(service):
    with pytest.raises(TextValidationError):
        service.humanize("", Style.natural, Intensity.medium)
