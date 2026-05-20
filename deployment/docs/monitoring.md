# Monitoring + Drift-Signal Strategy

Locked-in decisions for the deployment-side observability surface: which metrics are exposed at `/metrics`, which structured log events feed downstream drift detection, and which decisions sit explicitly out of scope for this portfolio.

## Decision Summary

| Concern | Choice |
|---------|--------|
| Metrics protocol | **Prometheus text exposition** at `/metrics` per service (port 8001 / 8002 / 8003) |
| Golden-signals coverage | **`http_requests_total` + `http_request_duration_seconds` + `http_requests_in_flight`** — 3 metrics covering all 4 signals |
| Per-model inference latency | **`model_inference_duration_seconds` Histogram** (single label: `model_name`, 5 bounded values) |
| Input-distribution drift signal | **Structured log event `input_distribution_sample`** sampled every Nth request per model |
| Drift-signal scope | **PCA + DNN routers only** — other endpoints lack a numeric feature vector to summarize |
| Sampling rate | **`INPUT_DISTRIBUTION_SAMPLE_EVERY`** env var (default 100; `0` disables) |
| Drift computation | **Out of scope** — the log lines are the upstream signal a real drift pipeline would consume |

## Four Golden Signals via the HTTP Metrics

Three Prometheus metrics defined in each service's `app/middleware/metrics.py` cover the SRE four golden signals at the HTTP layer:

| Signal | Metric | Type | Labels |
|--------|--------|------|--------|
| Latency | `http_request_duration_seconds` | Histogram | `method`, `path`, `status` |
| Traffic | `http_requests_total` | Counter | `method`, `path`, `status` |
| Errors | `http_requests_total` (filtered by `status` label) | Counter | (same as above) |
| Saturation | `http_requests_in_flight` | Gauge | `method`, `path` |

Together they answer the four operational questions Google's SRE book treats as the minimum viable signal set: how slow, how busy, how broken, how full.

### Cardinality

Every unique label-value combination becomes a separate time series in Prometheus storage. Bounded sets are safe; unbounded labels (user-id, request-body) blow up. The three metrics use only bounded labels:

- `method` — bounded by the HTTP-verb set (a handful in practice; under a dozen even counting WebDAV extensions)
- `status` — bounded by the IANA-registered HTTP status codes (fewer than 100)
- `path` — every endpoint in every service uses a static path, so `request.url.path` is safe directly. For a service that later adds path parameters (`/users/{id}`), the labeling needs to use `request.scope["route"].path` — the route template, not the instantiated path — otherwise cardinality grows unbounded.

### Bucket choice

The latency histogram uses buckets `(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)` in seconds. Finer than the prometheus-client default (which starts at 5ms) because PCA + DNN + Q-learning inference is sub-millisecond; the 5-second ceiling covers the TF Transformer translation worst-case. `histogram_quantile()` computes p50 / p95 / p99 across the bucket edges.

### Exclusions

The `/metrics` endpoint, `/openapi.json`, `/docs`, `/redoc`, and `/favicon.ico` are excluded from instrumentation. Without the exclusion, every Prometheus scrape would inflate the counters it is reading — `/metrics` would be the busiest "endpoint" by design.

### HEALTHCHECK noise on `/health`

Docker's HEALTHCHECK directive in each Dockerfile fires every 30 seconds and probes `http://127.0.0.1:<port>/health` via stdlib `urllib`. That probe hits the FastAPI middleware stack like any other request, so `http_requests_total{path="/health"}` accumulates roughly 120 hits per hour per container even with zero external traffic. For client-facing dashboards or alerts that should reflect real user traffic, filter out `path="/health"`:

```promql
sum(rate(http_requests_total{path!="/health"}[5m])) by (service)
```

`/ready` and `/health/<model>` are NOT probed by HEALTHCHECK — only `/health` is — so traffic on those endpoints reflects real callers (orchestrators or monitoring) with no HEALTHCHECK noise mixed in.

## Per-Model Inference Latency

The HTTP latency histogram above measures the FULL request: validation + preprocessing + inference + response serialization. That conflates "the model is slow" with "the request handler is slow." A separate metric isolates just the model-forward work:

```python
MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "model_inference_duration_seconds",
    "Model-forward inference latency in seconds (excludes request handling).",
    labelnames=["model_name"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
```

Single label `model_name` carries five bounded values across the portfolio: `sk-pca`, `pt-dnn`, `pt-gan-dcgan`, `pt-qlearning-taxi`, `tf-transformer-translation`. Adding any new model adds exactly one new series per bucket — safe for production storage.

Bucket floor is `0.0001s` (100 microseconds) — tighter than the HTTP histogram's `0.001s` because Q-learning lookups complete in microseconds, not milliseconds. Ceiling is `1.0s` — covers TF translation's greedy-decode worst-case while staying narrower than the HTTP `5.0s` ceiling.

### Wrap pattern

Each prediction router wraps its model-forward call in the histogram's `.time()` context manager:

```python
with MODEL_INFERENCE_DURATION_SECONDS.labels(model_name=MODEL_NAME).time():
    components_array = model.transform(X)
```

The context manager observes elapsed seconds on exit. Five wrap points across three services:

| Router | What is timed |
|--------|---------------|
| `sklearn-svc` `pca.py` | `pca.transform(X)` only |
| `pt-svc` `dnn.py` | The `no_grad` forward + softmax + argmax (model-output extraction) |
| `pt-svc` `gan.py` | The `no_grad` generator forward only (excludes denorm + PIL/PNG encode) |
| `pt-svc` `qlearning.py` | `Q[state]` index + argmax over 6 floats |
| `tf-svc` `translation.py` | Full encode + greedy-decode loop as one observation per request |

Each wrap is narrower than the existing `t0 = time.perf_counter()` scopes some routers use for the response's `generation_time_ms` field — deliberate, so model-forward time is isolated from pre/post processing.

## Input Distribution Sampling (Drift Signal)

A structured log event emits whole-vector summary statistics of incoming requests at a configurable cadence. Downstream consumers (a real drift-detection pipeline, an analytics dashboard, a log-search query) read these events to detect distribution shift in client traffic over time.

### Event shape

```json
{
  "event": "input_distribution_sample",
  "model_name": "sk-pca",
  "sample_index": 100,
  "n_features": 784,
  "mean": 0.345678,
  "std": 0.123456,
  "min": 0.0,
  "max": 1.0,
  "l2_norm": 12.345678,
  "zero_fraction": 0.234567,
  "level": "info",
  "timestamp": "2026-05-17T00:21:09.098698Z"
}
```

All floats rounded to six decimals. One log line per sample is roughly 250 bytes. The stats are aggregates over the whole feature vector — not per-feature — keeping log volume bounded. Per-feature PSI input would push each log line into the 10-20 KB range (561 means + 561 stds for DNN, 784+784 for PCA), unnecessary for a "something shifted" smoke alarm.

### Sampling cadence

The `INPUT_DISTRIBUTION_SAMPLE_EVERY` env var controls N:

| `INPUT_DISTRIBUTION_SAMPLE_EVERY` | Behavior |
|-----------------------------------|----------|
| Unset | Default `100` — emits roughly 1% of requests per model |
| `100` | Same as default |
| `1` | Emits every request (debug mode; high log volume) |
| `0` | Disables emission entirely; counter still increments cheaply for re-enablement |
| Negative or non-integer | Quiet fallback to default `100` |

The counter is keyed by `model_name` (`dict[str, int]` in `_COUNTERS`), so each model logs every Nth of its OWN requests. A global counter would skew the cadence toward whichever endpoint received more traffic; the per-model split keeps the time series even per model.

### Why these endpoints, not the others

Two of five deployed endpoints carry the helper call:

| Endpoint | Input | Helper call? |
|----------|-------|---------------|
| `/predict/pca` | 784 floats (Fashion-MNIST pixels) | Yes |
| `/predict/dnn` | 561 floats (UCI HAR normalized features) | Yes |
| `/predict/gan/sample` | `n_samples` (1-16) + optional `seed` | No — no client-side feature vector; the latent `z` is sampled server-side |
| `/predict/qlearning/taxi` | `state` (integer 0-499) | No — single integer is not a distribution to summarize |
| `/translate` | `text` (string) + `max_length` (1-25) | No — text input needs token-length or vocabulary signals, not numeric-vector stats |

A middleware approach would have to path-match the two relevant endpoints + replay the JSON body. An explicit helper call from the two routers is one line per router and keeps the logic discoverable.

### Where the helper lives

```text
services/sklearn-svc/app/middleware/input_distribution.py
services/pt-svc/app/middleware/input_distribution.py
```

Two copies, byte-identical. Matches the established per-service middleware copy pattern (`metrics.py`, `logging.py`, `request_id.py` all duplicate the same way) — each service is its own Python project with its own venv and lock file, so a shared helper module would couple them in a way they otherwise avoid.

The helper is called from each router right after the loader cache pull and BEFORE any preprocessing. Logging the RAW input matches what drift detection cares about — client-side distribution shift, not transformations applied server-side.

## What This Enables for Drift Detection

Two standard techniques operate on the log lines above:

- **Population Stability Index (PSI)** — measures distribution shift between two time windows. Real PSI requires per-feature histograms; the aggregate stats here surface coarser "something shifted" signals (mean drift, std drift, sparsity drift) that motivate a deeper per-feature look.
- **Kolmogorov-Smirnov test** — distribution-equality test between two samples. Operates on the raw values; our log lines provide summary statistics only, sufficient for triggering a sampled re-pull of raw values for the test.

Both techniques live on the consumer side. **The actual drift computation is out of scope for this portfolio** — what ships is the upstream signal a real drift system would consume. Building the consumer (window aggregator + PSI calculator + alert sink) is a follow-up project that would read the structured log stream from a log aggregator (Loki, CloudWatch Logs, Datadog Logs) rather than this codebase.

## Operating the Surface

### Scraping `/metrics` directly

```powershell
# Hit any service's metrics endpoint
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8001/metrics |
    Select-Object -ExpandProperty Content
```

The response is Prometheus text exposition format — series-per-line, plus `# HELP` and `# TYPE` comments. A scraper polls this endpoint on a configurable interval (15s is the Prometheus default) and persists the values as time series.

### Sample PromQL queries

Once a Prometheus instance is scraping the services, these queries answer the typical operator questions:

```promql
# Request rate per endpoint over the last 5 minutes (req/sec)
rate(http_requests_total[5m])

# 95th percentile request latency per endpoint over 5 minutes
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate (4xx + 5xx) as a fraction of total requests per endpoint
sum by (path) (rate(http_requests_total{status=~"4..|5.."}[5m]))
  / sum by (path) (rate(http_requests_total[5m]))

# 99th percentile model-only inference latency per model over 5 minutes
histogram_quantile(0.99, rate(model_inference_duration_seconds_bucket[5m]))

# Concurrent in-flight requests right now per endpoint
http_requests_in_flight

# Inference call volume per model over the last hour
increase(model_inference_duration_seconds_count[1h])
```

### Tailing structured logs

Every service emits JSON-formatted log lines via structlog. `jq` filters them to the events of interest:

```bash
# Watch every input-distribution sample across the stack
docker compose logs -f sklearn-svc pt-svc |
    jq -c 'select(.event == "input_distribution_sample")'

# Watch only request_finished events with non-2xx status
docker compose logs -f |
    jq -c 'select(.event == "request_finished" and .status >= 400)'
```

PowerShell equivalent for a single service:

```powershell
docker compose logs -f sklearn-svc |
    Select-String -Pattern '"event":"input_distribution_sample"'
```

### Tuning the sampling rate

For canary debugging — temporarily emit every request:

```powershell
docker run --rm -d -p 8001:8001 `
  -e INPUT_DISTRIBUTION_SAMPLE_EVERY=1 `
  ml-fc-sklearn-svc:dev
```

For high-traffic production where 1% sampling is still too much:

```yaml
# docker-compose.yml
environment:
  INPUT_DISTRIBUTION_SAMPLE_EVERY: 1000   # 0.1% sampling
```

For an environment that should not emit drift signals at all (e.g., a development container avoiding log noise):

```yaml
environment:
  INPUT_DISTRIBUTION_SAMPLE_EVERY: 0
```

## Failure Modes + Diagnostics

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/metrics` returns 503 | Service is not healthy (lifespan load failed) | Check `docker logs <container>`; the loader's chained exception will be in the traceback |
| `http_requests_total` shows zero series for an endpoint | No request has hit that endpoint since startup | Send one test request; the labeled series materializes on the first observation |
| `model_inference_duration_seconds_count` is zero for one model | No successful prediction has run for that model yet | Send a real prediction request through that router; verify the response is 200 |
| `input_distribution_sample` log lines never appear | `INPUT_DISTRIBUTION_SAMPLE_EVERY=0`, OR traffic has not reached N yet, OR the router is not calling the helper | Check the env var; if set correctly, send N requests through the router; if still nothing, grep the router source for the helper call |
| Sample log lines appear at irregular cadence | Multi-worker uvicorn — each worker has its own counter | Expected; aggregate signal still holds across workers. Use single-worker mode if even cadence matters more than throughput |
| `mean` / `std` / `l2_norm` values look NaN or implausible | Empty `features` array reached the helper | Should not happen in practice (Pydantic enforces `min_length` on the schemas); if it does, inspect upstream validation |
| Prometheus shows series with high cardinality | A new endpoint added a path parameter and is exposing instantiated paths instead of the route template | Switch the middleware to use `request.scope["route"].path` for that endpoint |

## What This Document Does Not Cover

- **Alerting rules** — translating the PromQL queries above into an AlertManager configuration. Out of scope; would live in an `alerting.md` once a Prometheus instance is provisioned.
- **Dashboards** — panel definitions for the metrics + log queries in a dashboarding tool. Out of scope; the metric names and log event shapes here are the primitives a future dashboard would build on.
- **Log aggregation backend** — wherever the JSON-stdout lines get collected and indexed. The services emit; how those lines flow downstream is a deployment-environment choice, not a service-design choice.
- **Real drift computation** — windowed aggregation of `input_distribution_sample` log lines, PSI / KS calculation, and threshold-based alerting. The log signal is the input; the computation pipeline is a follow-up project.
- **Per-model alias granularity in metrics** — the `model_name` label maps 1:1 to the loader's `MODEL_NAME` constant. Distinguishing `sk-pca@production` from `sk-pca@canary` in metrics would require either an additional `alias` label or a runtime label-mutation hook. Not implemented today; the operator distinguishes via the structured log line that records the resolved alias at load time (`<svc>_alias_resolved` event from `registry-strategy.md`).
- **Tracing** — distributed tracing for cross-service request flows. The `X-Request-ID` middleware lays the groundwork (every request gets a UUID, propagatable across services) but no trace exporter is wired up.
