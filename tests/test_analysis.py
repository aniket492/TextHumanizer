from __future__ import annotations

from ml.evaluation import metrics as m


def test_analyze_endpoint_structure(client, sample_text):
    resp = client.post("/api/analyze", json={"text": sample_text})
    assert resp.status_code == 200
    body = resp.json()
    for key in ("word_count", "sentence_count", "readability_score", "vocabulary_diversity", "repetition_score",
                "paragraph_count", "average_sentence_length", "punctuation", "repeated_words", "repeated_phrases"):
        assert key in body
    assert body["word_count"] == len(m.words(sample_text))
    assert body["sentence_count"] == 6
    assert body["paragraph_count"] == 1
    assert 0 < body["vocabulary_diversity"] <= 1
    assert body["semantic_similarity"] is None


def test_analyze_with_comparison(client):
    a = "The committee approved the new budget on Monday after a long debate."
    b = "After a long debate, the committee approved the new budget on Monday."
    c = "Penguins are flightless birds that live mostly in the Southern Hemisphere."
    same = client.post("/api/analyze", json={"text": a, "compare_text": b}).json()
    diff = client.post("/api/analyze", json={"text": a, "compare_text": c}).json()
    assert same["semantic_similarity"] > diff["semantic_similarity"]
    assert same["similarity_method"]


def test_analyze_validation(client):
    assert client.post("/api/analyze", json={"text": ""}).json()["error"]["code"] == "empty_input"
    assert client.post("/api/analyze", json={}).status_code == 422
    resp = client.post("/api/analyze", content=b"{not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_json"


def test_repetition_and_diversity_metrics():
    repetitive = "The model is good. The model is good. The model is good and the model is fast."
    varied = "Rain fell over the hills while farmers checked fences and children ran inside laughing."
    assert m.repetition_score(repetitive) > m.repetition_score(varied)
    assert m.vocabulary_diversity(varied) > m.vocabulary_diversity(repetitive)
    phrases = [p["text"] for p in m.repeated_phrases(repetitive)]
    assert any("model is good" in p for p in phrases)


def test_entity_and_factual_checks():
    src = "In 2021, NASA spent $2.5 billion on 14 missions, according to https://nasa.gov/budget [3]."
    good = "According to https://nasa.gov/budget [3], NASA spent $2.5 billion on 14 missions in 2021."
    bad = "NASA spent about $3 billion on 12 missions in 2020."
    score_good, missing_good = m.entity_preservation(src, good)
    score_bad, missing_bad = m.entity_preservation(src, bad)
    assert score_good == 1.0 and not missing_good
    assert score_bad < 1.0 and "https://nasa.gov/budget" in missing_bad
    assert m.factual_consistency(src, good) == 1.0
    assert m.factual_consistency(src, bad) < 0.6
    assert m.factual_consistency("The test did not pass.", "The test passed.") < 1.0


def test_grammar_and_readability():
    clean_score, _ = m.grammar_check("This sentence is fine. So is this one, which has a clause.")
    bad_score, issues = m.grammar_check("this is is broken (and unbalanced. a apple fell")
    assert clean_score > bad_score and issues
    easy = m.readability("The cat sat. It was warm. We had fun.")["flesch_reading_ease"]
    hard = m.readability("Institutional heterogeneity substantially complicates comparative epistemological evaluation.")["flesch_reading_ease"]
    assert easy > hard


def test_sentence_splitter_handles_abbreviations():
    s = m.split_sentences("Dr. Smith met Mr. J. Doe at 5 p.m. on Jan. 3. They talked, e.g. about U.S. policy. Done!")
    assert len(s) == 3
