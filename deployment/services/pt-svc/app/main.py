"""
pt-svc - FastAPI application entry point.

Hosts the PyTorch deployment endpoints:
    - D2: /predict/dnn          (UCI HAR activity classification)
    - D3: /predict/gan/sample   (DCGAN image generation - later phase)
    - D4: /predict/qlearning/taxi (Q-table action lookup - later phase)

USAGE (from deployment/services/pt-svc/):
    .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8002

Port convention across services:
    sklearn-svc -> 8001
    pt-svc      -> 8002    (this service)
    tf-svc      -> 8003

WHAT THIS FILE CONTAINS NOW:
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - /health (liveness) + /ready (readiness, currently always returns
      placeholder until later steps wires in the model loader)

WHAT WILL BE ADDED LATER:
    - Step 2.4: dnn_loader + lifespan event; /ready returns 503 until loaded
    - Step 2.5: include POST /predict/dnn router
    - Step 2.6: middleware stack (request_id + logging + metrics) -
      copied from sklearn-svc; framework-agnostic
"""

from fastapi import FastAPI

app = FastAPI(
    title="pt-svc",
    version="0.1.0",
    description=(
        "Deployment service for PyTorch models: D2 DNN (UCI HAR activity "
        "recognition, 96.03% test accuracy), D3 DCGAN (CIFAR-10 image "
        "generation, FID 30.57), D4 Q-Learning V1 Tabular (Taxi-v4). "
        "Loads from the consolidated MLflow registry at "
        "`deployment/mlflow.db`. Currently scaffolding - first endpoint "
        "(POST /predict/dnn) lands at Step 2.5."
    ),
)


# Health check endpoints
# Same liveness/readiness pattern as sklearn-svc - see that service's
# main.py for the full Kubernetes-probe rationale. /health stays cheap
# (no work); /ready will gate on model load completion once Step 2.4
# wires the lifespan event.


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """
    Liveness check - the process is alive and responding.

    Returns:
        {"status": "alive"} with HTTP 200.
    """
    return {"status": "alive"}


@app.get("/ready", tags=["health"])
async def ready() -> dict[str, str]:
    """
    Readiness check - the service can accept traffic.

    Returns:
        Placeholder until Step 2.4 (lifespan + dnn_loader). Will return
        503 with detail="model_not_loaded" until the lifespan event
        finishes loading the DNN + scaler.
    """
    return {"status": "ready", "model_loaded": "not yet (Step 2.4 adds this)"}
