from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from config import settings
from core.security import decode_access_token
from database import get_db
from models import User
from services.humanizer_service import HumanizerService
from services.model_service import ModelService, get_model_service

bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired authentication token.",
    headers={"WWW-Authenticate": "Bearer"},
)


def _user_from_credentials(credentials: HTTPAuthorizationCredentials | None, db: Session) -> User | None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    subject = decode_access_token(credentials.credentials)
    if subject is None or not subject.isdigit():
        raise _UNAUTHORIZED
    user = db.get(User, int(subject))
    if user is None:
        raise _UNAUTHORIZED
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = _user_from_credentials(credentials, db)
    assert user is not None
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    return _user_from_credentials(credentials, db)


def get_models() -> ModelService:
    return get_model_service()


@lru_cache
def _humanizer() -> HumanizerService:
    return HumanizerService(settings, get_model_service())


def get_humanizer() -> HumanizerService:
    return _humanizer()
