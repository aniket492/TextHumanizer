"""Password hashing (bcrypt) and JWT access tokens (PyJWT)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from config import settings

# bcrypt only uses the first 72 bytes of a password; we reject longer passwords at validation.
BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str | int, expires_minutes: int | None = None) -> tuple[str, int]:
    minutes = expires_minutes or settings.access_token_expire_minutes
    now = datetime.now(timezone.utc)
    payload = {"sub": str(subject), "iat": now, "exp": now + timedelta(minutes=minutes), "type": "access"}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, minutes * 60


def decode_access_token(token: str) -> str | None:
    """Return the subject (user id) of a valid token, or None."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError:
        return None
    if payload.get("type") != "access":
        return None
    return payload.get("sub")
