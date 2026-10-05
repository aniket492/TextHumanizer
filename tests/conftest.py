"""Test configuration.

Uses a throw-away SQLite database by default; set TEST_DATABASE_URL to run against
PostgreSQL. The rule-based engine and TF-IDF similarity are used so tests are fast
and need no model downloads (both are real implementations, not mocks).
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_tmp_db = Path(tempfile.gettempdir()) / f"humanizer_test_{uuid.uuid4().hex[:8]}.db"
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{_tmp_db.as_posix()}"
os.environ["JWT_SECRET"] = "test-secret-" + uuid.uuid4().hex
os.environ["ENVIRONMENT"] = "test"
os.environ["MODEL_BACKEND"] = os.environ.get("TEST_MODEL_BACKEND", "baseline")
os.environ["SIMILARITY_MODEL"] = os.environ.get("TEST_SIMILARITY_MODEL", "")
os.environ["MODEL_PRELOAD"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

from fastapi.testclient import TestClient  # noqa: E402

from database import Base, engine  # noqa: E402
import models  # noqa: E402,F401
from app import app  # noqa: E402

SAMPLE_TEXT = (
    "In today's fast-paced world, it is important to note that technology plays a crucial role in "
    "education. Furthermore, students utilize a wide range of digital tools in order to learn more "
    "effectively. Moreover, a 2023 survey by the National Center for Education Statistics found that 87% "
    "of schools use online platforms (Smith, 2023). Additionally, teachers can leverage these tools to "
    "personalize instruction. In conclusion, it is evident that technology will continue to shape the "
    "future of learning; schools that do not adapt will fall behind, and this will have long-term effects "
    "on students. More details are available at https://example.org/report."
)


@pytest.fixture(scope="session", autouse=True)
def _database():
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
    engine.dispose()
    if _tmp_db.exists():
        try:
            _tmp_db.unlink()
        except OSError:
            pass


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def register_user(client: TestClient, password: str = "Str0ngPassw0rd") -> tuple[dict, dict]:
    email = f"user_{uuid.uuid4().hex[:10]}@example.com"
    resp = client.post("/api/auth/register", json={"name": "Test User", "email": email, "password": password})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, {"email": email, "password": password, **body}


@pytest.fixture()
def auth(client) -> tuple[dict, dict]:
    return register_user(client)


@pytest.fixture()
def sample_text() -> str:
    return SAMPLE_TEXT
