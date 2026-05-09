"""
Prometheus metrics instrumentation.

WHAT THIS FILE PROVIDES:
    1. Three module-level metric objects (Counter, Histogram, Gauge) that
       become time series in Prometheus.
    2. `MetricsMiddleware` - per-request observer that increments the
       Counter, records latency in the Histogram, and tracks in-flight
       requests in the Gauge.

    The /metrics HTTP endpoint itself lives in main.py - it's a single
    one-liner that doesn't justify its own file, and it imports from
    prometheus_client directly.

WHY PROMETHEUS:
    The de-facto standard for service metrics in 2026. Pull-based:
    Prometheus scrapes our /metrics endpoint every 15s (configurable),
    stores the time series, and Grafana / Alertmanager build on top.

WHY THESE THREE METRICS:
    Standard HTTP service set used by every production team I've seen:
        - http_requests_total       (Counter)   - throughput, error rate
        - http_request_duration_secs(Histogram) - p50/p95/p99 latency
        - http_requests_in_flight   (Gauge)     - concurrency / saturation
    Together they answer the four golden signals (Google SRE book):
    latency (Histogram), traffic (Counter), errors (Counter w/ status
    label), saturation (Gauge).

LABEL CARDINALITY:
    Each unique combination of label values is a separate time series in
    Prometheus storage. Bounded sets are safe (HTTP method = ~5 verbs,
    status = ~40 codes); unbounded labels (user_id, request body) blow
    up storage. Every endpoint in this service uses a static path, so
    using request.url.path directly is safe. For services with path
    parameters (e.g., /users/{id}), the labeling needs to use
    request.scope["route"].path - the template, not the instantiated
    path - otherwise label cardinality grows unbounded.

EXCLUSIONS:
    The /metrics endpoint itself is NOT instrumented. Otherwise every
    Prometheus scrape would inflate the counters - the metrics endpoint
    would always be the busiest "endpoint" by design.
"""

import time
from collections.abc import Awaitable, Callable

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Metric definitions
"""
These are MODULE-LEVEL objects on purpose. The prometheus_client library
uses a global registry; defining the metric once at import time means
every middleware instance and every test shares the same series.
Re-creating a metric with the same name would raise.

Latency histogram buckets (in SECONDS).
Tuned for fast inference services (single-digit ms typical):
  - Fine resolution at the low end where most requests live
  - Wider buckets at the high end to catch slow outliers
Prometheus default starts at 5ms; we go finer because the DNN
inference is sub-millisecond. Bucket boundaries are inclusive
("le" = "less or equal").
"""
HTTP_LATENCY_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
)

# Counter: monotonically increasing count of HTTP requests.
# Prometheus computes rate() over windows: rate(http_requests_total[5m])
# = requests per second over the last 5 minutes.
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received.",
    labelnames=["method", "path", "status"],
)

# Histogram: latency distribution. Prometheus stores per-bucket counts
# (`http_request_duration_seconds_bucket{le="0.05"}` = "how many requests
# completed in <= 50ms"). histogram_quantile() gives us p50/p95/p99.
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=["method", "path", "status"],
    buckets=HTTP_LATENCY_BUCKETS,
)

# Gauge: instantaneous count of in-progress requests. Spikes here when
# the service is saturated (handler slower than incoming RPS). Note:
# Gauges aren't labeled by `status` because we don't know the status
# until the request finishes - the in-flight count is "started but not
# yet finished" by definition.
HTTP_REQUESTS_IN_FLIGHT = Gauge(
    "http_requests_in_flight",
    "HTTP requests currently being processed.",
    labelnames=["method", "path"],
)

# Paths we DON'T instrument. /metrics is the obvious one (avoid the
# scraper inflating its own counters). /openapi.json and /docs / /redoc
# are also dev/inspection traffic, not real workload.
EXCLUDED_PATHS: frozenset[str] = frozenset(
    {"/metrics", "/openapi.json", "/docs", "/redoc", "/favicon.ico"}
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Per-request Prometheus instrumentation.

    Increments HTTP_REQUESTS_TOTAL, observes HTTP_REQUEST_DURATION_SECONDS,
    and tracks HTTP_REQUESTS_IN_FLIGHT for every request EXCEPT those
    matching EXCLUDED_PATHS.

    Order: this middleware should be INNERMOST (registered FIRST in
    main.py) so the latency it measures is purely handler+inner-middleware
    time, not including the outer request_id and logging overhead. We're
    measuring the work, not the bookkeeping.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path

        # Skip excluded paths - just pass through without instrumentation.
        if path in EXCLUDED_PATHS:
            return await call_next(request)

        method = request.method

        # Increment the in-flight gauge BEFORE running the handler;
        # decrement AFTER (in finally so it always runs even on exception).
        # Using .labels(...) creates the labeled child series the first
        # time the combo appears.
        in_flight = HTTP_REQUESTS_IN_FLIGHT.labels(method=method, path=path)
        in_flight.inc()

        # Default in case a pathological exception happens before we've
        # assigned a real status. Exception handlers below override.
        status = "500"
        start = time.perf_counter()

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            # Even if the handler raises, we want the metric to count it.
            # Use status="500" since FastAPI's default exception handler
            # converts unhandled exceptions to 500. Re-raise so FastAPI's
            # error pipeline still runs.
            status = "500"
            raise
        finally:
            duration_seconds = time.perf_counter() - start
            in_flight.dec()
            # Counter increment + Histogram observation share the same
            # label set. We do them in the finally so they record even
            # on exception paths (the `raise` above propagates AFTER
            # this finally block runs).
            HTTP_REQUESTS_TOTAL.labels(
                method=method, path=path, status=status
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method, path=path, status=status
            ).observe(duration_seconds)

        return response
