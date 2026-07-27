"""Health and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from src.api import state as service_state
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    """Report service status and which model is currently served."""
    current = service_state.state
    metadata = current.engine.bundle.metadata if current.engine is not None else None
    return HealthResponse(
        status="ok" if current.model_loaded else "degraded",
        model_loaded=current.model_loaded,
        model_name=metadata.model_name if metadata else None,
        model_version=metadata.version if metadata else None,
        version=current.config.app.version,
        uptime_seconds=round(current.uptime_seconds, 3),
    )


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus-style metrics")
def metrics() -> str:
    """Expose counters in the Prometheus text exposition format."""
    current = service_state.state
    processor = current.processor
    lifetime = processor.lifetime_stats() if processor is not None else {}
    severity = current.alert_manager.severity_counts() if current.alert_manager else {}

    lines = [
        "# HELP nids_model_loaded Whether a detection model is loaded.",
        "# TYPE nids_model_loaded gauge",
        f"nids_model_loaded {int(current.model_loaded)}",
        "# HELP nids_uptime_seconds Service uptime in seconds.",
        "# TYPE nids_uptime_seconds gauge",
        f"nids_uptime_seconds {current.uptime_seconds:.3f}",
        "# HELP nids_flows_processed_total Flows processed since start-up.",
        "# TYPE nids_flows_processed_total counter",
        f"nids_flows_processed_total {lifetime.get('total_processed', 0)}",
        "# HELP nids_attacks_detected_total Attacks detected since start-up.",
        "# TYPE nids_attacks_detected_total counter",
        f"nids_attacks_detected_total {lifetime.get('total_attacks', 0)}",
        "# HELP nids_alerts_total Alerts dispatched by severity.",
        "# TYPE nids_alerts_total counter",
    ]
    lines += [f'nids_alerts_total{{severity="{name}"}} {count}' for name, count in severity.items()]
    return "\n".join(lines) + "\n"
