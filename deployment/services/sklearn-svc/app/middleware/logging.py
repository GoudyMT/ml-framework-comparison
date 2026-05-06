"""
Structured logging configuration + per-request logging middleware.

WHAT THIS FILE PROVIDES:
    1. `configure_logging()` - one-time setup of structlog + stdlib logging.
       Call once at app boot (before any logger is created).
    2. `LoggingMiddleware` - a BaseHTTPMiddleware that emits a JSON log
       line at request start and at request finish, enriched with
       request_id, method, path, status, duration_ms.

WHY STRUCTURED LOGS:
    Free-form text logs are searchable by substring; structured (JSON)
    logs are queryable by field. Production log aggregators (CloudWatch,
    Datadog, Loki) auto-index every JSON field, so questions like
    "show me all 5xx on /predict/pca slower than 100ms in the last hour"
    become single queries instead of regex archaeology.

WHY STRUCTLOG (vs raw stdlib `logging`):
    stdlib `logging` produces text. You can layer JSON formatters onto
    it, but adding context (request_id, user_id) means threading kwargs
    through every call site. structlog's contextvars-based design lets
    you `bind_contextvars(request_id=...)` ONCE per request and every
    subsequent log line in that async task inherits the binding.

    structlog also wraps stdlib so OTHER libraries (uvicorn, mlflow,
    sklearn) emit through the same JSON pipeline. Single output stream,
    consistent format.

DESIGN DECISIONS LOCKED:
    - JSON renderer in BOTH dev and prod. Matching formats means dev
      logs surface real production issues (e.g., a field that doesn't
      serialize cleanly).
    - ISO 8601 UTC timestamps. Universal, sortable, no timezone drift.
    - We do NOT suppress uvicorn's access log. It'll flow through the
      stdlib bridge and come out as JSON too. Some redundancy with our
      middleware logs, but our logs add request_id + duration.
"""

import logging
import sys
import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from structlog.typing import Processor


def configure_logging(level: str = "INFO") -> None:
    """
    Configure structlog + stdlib logging to emit JSON to stdout.

    Idempotent - safe to call multiple times (e.g., from tests). Runs
    once at app boot in main.py BEFORE any module-level logger is created.

    Args:
        level: Log level threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Default INFO matches production - DEBUG is too noisy for
            prod and slows things down.

    What this does:
        1. Configures structlog with a processor pipeline:
           - merge_contextvars: pull contextvars (request_id, etc.) into
             the log entry. MUST be first so subsequent processors see them.
           - add_log_level: add "level" field
           - TimeStamper: add ISO 8601 UTC "timestamp" field
           - format_exc_info: render exceptions as a structured field
             instead of printing the traceback as a separate line
           - JSONRenderer: serialize the dict to a JSON string
        2. Routes stdlib logging (uvicorn, mlflow, etc.) through the
           same JSON renderer via ProcessorFormatter. Foreign-library
           logs come out the same shape as our own.
    """
    # Step 1: structlog's processor pipeline.
    # Order matters - each processor sees the dict the previous one built.
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    structlog.configure(
        processors=[
            # contextvars must come FIRST so request_id, etc. are visible
            # to every subsequent processor.
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        # BoundLogger is the user-facing logger class. Supports .bind()
        # for inline context attachment in addition to contextvars.
        wrapper_class=structlog.stdlib.BoundLogger,
        # LoggerFactory backs structlog with stdlib's logging - lets stdlib
        # config (handlers, levels) control structlog output.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Cache the logger built on first use - performance optimization.
        cache_logger_on_first_use=True,
    )

    # Step 2: stdlib logging bridge for foreign libraries.
    # Explicit type annotation: mypy infers a heterogeneous list (mix of
    # functions and a TimeStamper instance) as list[object]; ProcessorFormatter
    # wants Sequence[Processor]. Annotating with structlog.typing.Processor
    # gives mypy the right type without changing runtime behavior.
    foreign_pre_chain: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
    ]
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=foreign_pre_chain,
        )
    )

    root = logging.getLogger()
    # Replace existing handlers - prevents double-logging if uvicorn or
    # something else already configured stdlib logging.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Re-route uvicorn's pre-configured loggers through our root handler.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request log emitter.

    Logs at two points:
        - `request_started`: when the request enters this middleware
        - `request_finished`: after the handler returns

    Both lines automatically include `request_id` (bound into contextvars
    so it propagates to ANY log call inside the request, not just ours).

    Order: register this AFTER RequestIDMiddleware in main.py - because
    add_middleware order is reverse-of-execution, that means request_id
    is the OUTERMOST layer (runs first, sees raw request), and we run
    inside it (so request.state.request_id is already populated).
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Pull the request_id placed by RequestIDMiddleware. getattr with
        # default "-" guards against the unlikely case where this
        # middleware runs before request_id (misconfigured order).
        request_id = getattr(request.state, "request_id", "-")

        # bind_contextvars attaches request_id to structlog's per-task
        # context. Every structlog.get_logger().info(...) call in this
        # request - including from pca_loader and the router - will
        # automatically include {"request_id": "..."} in the JSON output.
        structlog.contextvars.bind_contextvars(request_id=request_id)

        log = structlog.get_logger("http")

        try:
            log.info(
                "request_started",
                method=request.method,
                path=request.url.path,
                client=request.client.host if request.client else None,
            )

            # perf_counter is the right clock for measuring durations -
            # monotonic, high resolution, immune to wall-clock adjustments.
            start = time.perf_counter()
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000.0

            log.info(
                "request_finished",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            return response
        finally:
            # CRITICAL: clear contextvars at request boundary. Otherwise
            # the next request reusing this task could see a stale
            # request_id. async tasks are pooled, contextvars persist.
            structlog.contextvars.clear_contextvars()
