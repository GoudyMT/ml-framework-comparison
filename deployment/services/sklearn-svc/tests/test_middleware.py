"""
Tests for the middleware stack (request_id + metrics).

WHAT THESE TESTS COVER:
    Request ID middleware:
        - No client header -> server generates a fresh UUID, echoes it
          in X-Request-ID response header.
        - Client sends a valid UUID -> server honors it (distributed
          tracing path).
        - Client sends garbage -> server REJECTS and generates fresh
          (defense against log poisoning).

    /metrics endpoint:
        - Returns 200 with text/plain content-type (Prometheus exposition).
        - Body contains the three core metric names we defined.
        - /metrics itself is excluded from instrumentation (no
          http_requests_total entry for path="/metrics").

WHAT THESE TESTS DO NOT COVER:
    - Logging middleware: structlog output is hard to assert against
      cleanly (it goes to stdout via the stdlib bridge). Manually
      verified end-to-end in 1.6b. Could capture stdout in a future
      pass if it matters.
    - Real Prometheus scraping (out of scope for unit tests).
"""

import uuid

from fastapi.testclient import TestClient

# Constants


REQUEST_ID_HEADER = "X-Request-ID"


def _is_valid_uuid(value: str) -> bool:
    """Mirror of the production validator - any RFC 4122 UUID."""
    try:
        uuid.UUID(value)
    except (ValueError, TypeError):
        return False
    return True


# Request ID middleware


def test_request_id_generated_when_absent(client: TestClient) -> None:
    """
    No X-Request-ID header sent -> server generates a fresh valid UUID.

    Hits /health (cheapest endpoint) just to flow through middleware.
    """
    response = client.get("/health")

    assert response.status_code == 200
    rid = response.headers.get(REQUEST_ID_HEADER)
    assert rid is not None, "Server must always set X-Request-ID"
    assert _is_valid_uuid(rid), f"Generated ID isn't a valid UUID: {rid!r}"


def test_request_id_echoed_when_client_sends_valid_uuid(
    client: TestClient,
) -> None:
    """
    Client sends a valid UUID -> server echoes it back unchanged.

    This is the distributed-tracing path: an upstream service passes its
    own request_id, sklearn-svc adopts it so logs across services can
    be correlated by one ID.
    """
    client_id = "11111111-2222-3333-4444-555555555555"
    response = client.get(
        "/health", headers={REQUEST_ID_HEADER: client_id}
    )

    assert response.status_code == 200
    assert response.headers.get(REQUEST_ID_HEADER) == client_id


def test_request_id_replaced_when_client_sends_garbage(
    client: TestClient,
) -> None:
    """
    Client sends a non-UUID string -> server REJECTS it, generates fresh.

    Defense against log poisoning: a malicious client could otherwise
    set X-Request-ID="GET-/admin-from-internal-script" and have that
    string appear in our logs as if it were a legitimate ID.
    """
    garbage = "definitely-not-a-uuid-haha"
    response = client.get(
        "/health", headers={REQUEST_ID_HEADER: garbage}
    )

    assert response.status_code == 200
    rid = response.headers.get(REQUEST_ID_HEADER)
    assert rid is not None
    assert rid != garbage, "Server accepted garbage instead of regenerating"
    assert _is_valid_uuid(rid), f"Replacement ID isn't a valid UUID: {rid!r}"


# /metrics endpoint


def test_metrics_endpoint_returns_prometheus_format(
    client: TestClient,
) -> None:
    """
    /metrics returns 200 with the Prometheus text exposition format.

    Prometheus scrapers expect Content-Type "text/plain; version=0.0.4;
    charset=utf-8". The body is a series of HELP/TYPE/value lines, not
    JSON.
    """
    response = client.get("/metrics")

    assert response.status_code == 200
    # The exact content-type string is "text/plain; version=0.0.4;
    # charset=utf-8". We assert on the prefix so version bumps in the
    # prometheus_client library don't break us.
    assert response.headers["content-type"].startswith("text/plain")


def test_metrics_endpoint_exposes_our_metrics(client: TestClient) -> None:
    """
    The exposition body contains the three metrics we registered.

    We fire one /health to make sure the counters have at least one
    sample for the labels to materialize, then scrape /metrics.
    """
    # Generate one observation so the histogram + counter have non-empty
    # series (otherwise prometheus_client only emits HELP/TYPE lines for
    # them). Without this, the bucket lines wouldn't appear yet.
    client.get("/health")

    body = client.get("/metrics").text

    # The three metric NAMES (Prometheus prefixes Counter+Histogram+Gauge
    # names with helpful TYPE/HELP comments; we just assert presence).
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body
    assert "http_requests_in_flight" in body


def test_metrics_endpoint_excluded_from_instrumentation(
    client: TestClient,
) -> None:
    """
    /metrics is in EXCLUDED_PATHS - scraping it does NOT increment
    http_requests_total{path="/metrics", ...}.

    Without this exclusion, every Prometheus scrape would inflate the
    very metric it's reading - /metrics would become the busiest
    "endpoint" by design.
    """
    # Hit /metrics a few times.
    for _ in range(3):
        client.get("/metrics")

    body = client.get("/metrics").text

    # Confirm there's no labeled series for /metrics. Search for the
    # specific labeled-counter line that WOULD be there if we instrumented.
    assert 'http_requests_total{method="GET",path="/metrics"' not in body
