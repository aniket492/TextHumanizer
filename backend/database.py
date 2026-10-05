"""SQLAlchemy engine, session factory and declarative base."""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


class Base(DeclarativeBase):
    pass


def _make_engine(url: str):
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=5, max_overflow=10, connect_args={"connect_timeout": 5})
    return create_engine(url, **kwargs)


engine = _make_engine(settings.database_url)

if engine.dialect.name == "sqlite":

    @event.listens_for(engine, "connect")
    def _sqlite_fk_pragma(dbapi_conn, _):  # enforce ON DELETE CASCADE like PostgreSQL
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
