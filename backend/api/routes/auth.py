from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from core.logging import get_logger
from core.security import create_access_token, hash_password, verify_password
from database import get_db
from models import Usage, User
from schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = get_logger("auth")

# Used to keep login timing similar whether or not the e-mail exists.
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password-1")


def _token_response(user: User) -> TokenResponse:
    token, expires_in = create_access_token(user.id)
    return TokenResponse(access_token=token, expires_in=expires_in, user=UserOut.model_validate(user))


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if db.scalar(select(User.id).where(User.email == payload.email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this e-mail already exists.")
    user = User(name=payload.name, email=payload.email, password_hash=hash_password(payload.password))
    user.usage = Usage(request_count=0, words_processed=0)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this e-mail already exists.")
    db.refresh(user)
    logger.info("User registered id=%s", user.id)
    return _token_response(user)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None:
        verify_password(payload.password, _DUMMY_HASH)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect e-mail or password.")
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect e-mail or password.")
    logger.info("User logged in id=%s", user.id)
    return _token_response(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(user)
