from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt

from config import settings
from core.security import create_access_token, decode_access_token, hash_password, verify_password
from tests.conftest import register_user


def test_password_hashing_roundtrip():
    h = hash_password("S3curePassword")
    assert h != "S3curePassword" and h.startswith("$2")
    assert verify_password("S3curePassword", h)
    assert not verify_password("wrong-password1", h)
    assert not verify_password("anything", "not-a-hash")


def test_register_returns_token_and_user(client):
    email = f"new_{uuid.uuid4().hex[:8]}@Example.com"
    resp = client.post("/api/auth/register", json={"name": "  Ada   Lovelace ", "email": email, "password": "Analytical1"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert body["user"]["email"] == email.lower()
    assert body["user"]["name"] == "Ada Lovelace"
    assert "password" not in str(body).lower().replace("password_hash", "")


def test_register_duplicate_email_rejected(client, auth):
    _, user = auth
    resp = client.post("/api/auth/register", json={"name": "Dup", "email": user["email"].upper(), "password": "Another1pass"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "conflict"


def test_register_validation_errors(client):
    resp = client.post("/api/auth/register", json={"name": "X", "email": "not-an-email", "password": "short"})
    assert resp.status_code == 422
    fields = {d["field"] for d in resp.json()["error"]["details"]}
    assert {"email", "password"} <= fields
    resp = client.post("/api/auth/register", json={"name": "X", "email": "a@example.com", "password": "lettersonly"})
    assert resp.status_code == 422


def test_login_success_and_failure(client, auth):
    _, user = auth
    ok = client.post("/api/auth/login", json={"email": user["email"], "password": user["password"]})
    assert ok.status_code == 200 and ok.json()["access_token"]
    bad = client.post("/api/auth/login", json={"email": user["email"], "password": "WrongPass123"})
    assert bad.status_code == 401
    missing = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "WrongPass123"})
    assert missing.status_code == 401
    assert bad.json()["error"]["message"] == missing.json()["error"]["message"]


def test_me_requires_valid_token(client, auth):
    headers, user = auth
    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200 and me.json()["email"] == user["email"]
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage.token.value"}).status_code == 401


def test_expired_and_tampered_tokens_rejected(client, auth):
    _, user = auth
    uid = user["user"]["id"]
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    expired = jwt.encode({"sub": str(uid), "iat": past, "exp": past + timedelta(minutes=1), "type": "access"},
                         settings.jwt_secret, algorithm=settings.jwt_algorithm)
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    forged = jwt.encode({"sub": str(uid), "exp": datetime.now(timezone.utc) + timedelta(hours=1), "type": "access"},
                        "attacker-secret-that-is-long-enough-0123456789", algorithm="HS256")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_token_for_deleted_or_unknown_user_rejected(client):
    token, _ = create_access_token(987654321)
    assert decode_access_token(token) == "987654321"
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_register_helper_creates_distinct_users(client):
    _, a = register_user(client)
    _, b = register_user(client)
    assert a["user"]["id"] != b["user"]["id"]
