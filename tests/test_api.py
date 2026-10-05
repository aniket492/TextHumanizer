from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from database import engine
from models import HumanizationHistory, Usage, User
from tests.conftest import BACKEND, register_user


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "AI Humanizer"
    assert resp.headers.get("X-Request-ID")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["model"]["backend"] == "baseline"


def test_unknown_route_structured_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_humanize_requires_auth(client, sample_text):
    resp = client.post("/api/humanize", json={"text": sample_text})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_humanize_validation(client, auth):
    headers, _ = auth
    assert client.post("/api/humanize", json={"text": ""}, headers=headers).json()["error"]["code"] == "empty_input"
    resp = client.post("/api/humanize", json={"text": "valid text here ok", "tone": "pirate"}, headers=headers)
    assert resp.status_code == 422 and resp.json()["error"]["details"][0]["field"] == "tone"
    resp = client.post("/api/humanize", json={"text": "valid text here ok", "intensity": "max"}, headers=headers)
    assert resp.status_code == 422
    long_text = "word " * 5000
    resp = client.post("/api/humanize", json={"text": long_text}, headers=headers)
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "input_too_long"


def test_humanize_history_usage_flow(client, auth, sample_text):
    headers, _ = auth
    resp = client.post("/api/humanize", json={"text": sample_text, "tone": "professional", "intensity": "strong"}, headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] and body["output_text"] and body["tone"] == "professional"
    assert body["quality"]["semantic_similarity"] > 0
    assert body["input_word_count"] > 50

    # Not saved to history when save_history is false, but still counted in usage.
    resp2 = client.post("/api/humanize", json={"text": sample_text, "save_history": False}, headers=headers)
    assert resp2.status_code == 200 and resp2.json()["id"] is None

    hist = client.get("/api/history", headers=headers).json()
    assert hist["total"] == 1 and hist["items"][0]["id"] == body["id"]
    detail = client.get(f"/api/history/{body['id']}", headers=headers).json()
    assert detail["output_text"] == body["output_text"]
    assert detail["quality"]["engine"] == body["engine"]

    usage = client.get("/api/usage", headers=headers).json()
    assert usage["request_count"] == 2
    assert usage["words_processed"] == 2 * body["input_word_count"]
    assert usage["history_count"] == 1

    assert client.delete(f"/api/history/{body['id']}", headers=headers).status_code == 204
    assert client.get(f"/api/history/{body['id']}", headers=headers).status_code == 404
    assert client.get("/api/history", headers=headers).json()["total"] == 0


def test_history_isolated_between_users(client, auth, sample_text):
    headers_a, _ = auth
    headers_b, _ = register_user(client)
    item = client.post("/api/humanize", json={"text": sample_text}, headers=headers_a).json()
    assert client.get(f"/api/history/{item['id']}", headers=headers_b).status_code == 404
    assert client.delete(f"/api/history/{item['id']}", headers=headers_b).status_code == 404
    assert client.get("/api/history", headers=headers_b).json()["total"] == 0
    assert client.get(f"/api/history/{item['id']}", headers=headers_a).status_code == 200


def test_history_pagination(client, auth):
    headers, _ = auth
    text = "Furthermore, it is important to note that the team utilizes many tools in order to ship faster."
    ids = [client.post("/api/humanize", json={"text": text}, headers=headers).json()["id"] for _ in range(3)]
    page1 = client.get("/api/history?page=1&page_size=2", headers=headers).json()
    page2 = client.get("/api/history?page=2&page_size=2", headers=headers).json()
    assert page1["total"] == 3 and len(page1["items"]) == 2 and len(page2["items"]) == 1
    assert [i["id"] for i in page1["items"]] == sorted(ids, reverse=True)[:2]
    assert client.get("/api/history?page=0", headers=headers).status_code == 422


def test_database_operations_cascade(auth):
    _, user = auth
    uid = user["user"]["id"]
    with Session(engine) as db:
        db.add(HumanizationHistory(user_id=uid, original_text="a", output_text="b", tone="natural", intensity="light", word_count=1))
        db.commit()
        assert db.scalar(select(Usage).where(Usage.user_id == uid)) is not None
        db.delete(db.get(User, uid))
        db.commit()
        assert db.scalars(select(HumanizationHistory).where(HumanizationHistory.user_id == uid)).all() == []
        assert db.scalar(select(Usage).where(Usage.user_id == uid)) is None


def test_alembic_migrations_upgrade_and_downgrade():
    db_file = Path(tempfile.gettempdir()) / f"alembic_{uuid.uuid4().hex[:8]}.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_file.as_posix()}"}
    try:
        for args in (["upgrade", "head"], ["downgrade", "base"], ["upgrade", "head"]):
            proc = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=BACKEND, env=env, capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr
        eng = create_engine(env["DATABASE_URL"])
        tables = set(inspect(eng).get_table_names())
        eng.dispose()
        assert {"users", "humanization_history", "usage"} <= tables
    finally:
        db_file.unlink(missing_ok=True)
