"""FastAPI application: wiring, middleware and error handling only (no ML code here)."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import settings
from core.logging import get_logger, setup_logging

setup_logging(settings.log_level)
logger = get_logger("app")

from api.routes import analysis, auth, health, history, humanize  # noqa: E402
from ml.inference.predictor import ModelTimeoutError  # noqa: E402
from services.model_service import ModelUnavailableError, get_model_service  # noqa: E402
from utils.validators import TextValidationError  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s (env=%s, model_backend=%s)", settings.app_name, settings.environment, settings.model_backend)
    if settings.model_preload:
        get_model_service().warm_up()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=health.VERSION,
    description="Rewrite mechanical text into natural writing while preserving meaning and key facts.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error rid=%s %s %s", request_id, request.method, request.url.path)
        response = _error(500, "internal_error", "An unexpected error occurred.")
    duration = (time.perf_counter() - start) * 1000
    # Only method, path, status and timing are logged: never bodies, tokens or user text.
    logger.info("rid=%s %s %s -> %d (%.0f ms)", request_id, request.method, request.url.path, response.status_code, duration)
    response.headers["X-Request-ID"] = request_id
    return response


def _error(status_code: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status_code, headers=headers)


_STATUS_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
                 409: "conflict", 413: "payload_too_large", 422: "validation_error", 429: "rate_limited"}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request, exc: StarletteHTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return _error(exc.status_code, _STATUS_CODES.get(exc.status_code, "error"), message, headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    errors = exc.errors()
    if any(e.get("type") == "json_invalid" for e in errors):
        return _error(400, "invalid_json", "The request body is not valid JSON.")
    details = [
        {"field": ".".join(str(p) for p in e.get("loc", []) if p != "body"), "message": e.get("msg", "Invalid value")}
        for e in errors
    ]
    return _error(422, "validation_error", "Some fields are invalid.", details)


@app.exception_handler(TextValidationError)
async def text_validation_handler(_: Request, exc: TextValidationError):
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(_: Request, exc: ModelUnavailableError):
    logger.error("Model unavailable: %s", exc)
    return _error(503, "model_unavailable", "The rewrite model is temporarily unavailable. Please try again shortly.")


@app.exception_handler(ModelTimeoutError)
async def model_timeout_handler(_: Request, exc: ModelTimeoutError):
    logger.error("Model timeout: %s", exc)
    return _error(504, "model_timeout", "The rewrite took too long. Try a shorter text or a lighter intensity.")


@app.exception_handler(OperationalError)
@app.exception_handler(DBAPIError)
async def database_error_handler(_: Request, exc: Exception):
    logger.error("Database error: %s", type(exc).__name__)
    return _error(503, "database_unavailable", "The database is temporarily unavailable.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", type(exc).__name__)
    return _error(500, "internal_error", "An unexpected error occurred.")


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(humanize.router)
app.include_router(analysis.router)
app.include_router(history.router)
