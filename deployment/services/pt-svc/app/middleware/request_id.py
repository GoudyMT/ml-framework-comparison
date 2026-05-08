"""
Request ID middleware.

WHAT THIS DOES:
    Tags every HTTP request with a unique ID, available three ways:
        1. As `request.state.request_id` for handlers + later middleware
        2. As an `X-Request-ID` header on the OUTGOING response
        3. (Step 1.6b) As a field in every log line for that request

WHY THIS EXISTS:
    When something goes wrong in production ("my request was slow at 2:13am"),
    you need to find THAT specific request's log lines among millions.
    Without an ID, you're guessing from timestamps and IPs. With an ID, you
    grep one string across the log system and get the full story.

    The same ID also lets you correlate across services: if sklearn-svc
    eventually calls pt-svc internally, propagating the ID through the
    HTTP call means the entire distributed flow is traceable from a single
    grep. We don't have multi-service flows yet, but we set up the
    infrastructure now.

DESIGN: TRUST-BUT-VERIFY
    If the client sends `X-Request-ID: <something>`, we'll honor it - but
    only if it parses as a real UUID. This gives us:
        - Distributed tracing (clients/upstream services pass IDs through)
        - Defense against log poisoning (a malicious client can't inject
          `X-Request-ID: GET-/admin-from-internal-script` into our logs)

    If no header is sent, or it's malformed, we generate a fresh UUID4.

WHY BASEHTTPMIDDLEWARE (vs @app.middleware("http")):
    BaseHTTPMiddleware is the reusable Starlette base class. Putting the
    middleware in its own file under app/middleware/ means:
        - Tests can import + test it without booting the whole app
        - main.py stays a wiring file, not a middleware definition file
    The @decorator approach is fine for trivial cases; we're not in one.
"""

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

"""
Header name we read incoming + write outgoing. The "X-" prefix is the
legacy convention for non-standard HTTP headers; it's still universal in
2026 and matches what AWS ALB / GCP / Cloudflare inject by default.
"""
REQUEST_ID_HEADER: str = "X-Request-ID"


def _is_valid_uuid(value: str) -> bool:
    """
    True if `value` parses as any RFC 4122 UUID (any version).

    We don't require version 4 specifically - UUID1, UUID5, etc. are all
    fine for correlation purposes. We just want SOMETHING that looks
    UUID-shaped so attackers can't inject arbitrary log strings.
    """
    try:
        uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a request_id to every HTTP request and the matching response.

    Order matters: this middleware should be the OUTERMOST one (registered
    LAST in main.py - FastAPI runs middleware in reverse registration
    order, so the last-added is the first to see incoming requests). That
    way every later middleware (logging, metrics) and every handler sees
    the same request_id we generated/validated here.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """
        Per-request entry point. Runs once per HTTP request.

        Args:
            request: The incoming Starlette Request (FastAPI Request inherits).
            call_next: Async callable that invokes the rest of the
                middleware chain + the handler. Returns the Response.

        Returns:
            The Response from `call_next`, with X-Request-ID header set.
        """
        # Pull a client-provided ID if it's there and valid; else generate.
        client_provided = request.headers.get(REQUEST_ID_HEADER)
        if client_provided and _is_valid_uuid(client_provided):
            request_id = client_provided
        else:
            # uuid4() = random 122-bit ID. Collisions are mathematically
            # negligible (would need ~10^18 IDs before 50% collision prob).
            # str() gives the canonical hyphenated lowercase form.
            request_id = str(uuid.uuid4())

        # Attach to request state - Starlette's per-request scratch object.
        # Handlers and downstream middleware can read this without parsing
        # the header again.
        request.state.request_id = request_id

        # Run the rest of the chain (other middleware + the handler).
        # Whatever Response comes back is what we send to the client,
        # after we add our header.
        response = await call_next(request)

        # Set the header on the OUTGOING response so the client (or an
        # upstream proxy) can log/correlate using the same ID.
        response.headers[REQUEST_ID_HEADER] = request_id

        return response
