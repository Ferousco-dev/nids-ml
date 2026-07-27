"""Detection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from src.api import state as service_state
from src.api.schemas import (
    AlertOutput,
    BatchDetectionResponse,
    BatchFlowInput,
    DetectionResponse,
    DetectionStats,
    FlowInput,
)
from src.detection.engine import DetectionResult
from src.utils.logger import get_logger
from src.utils.validators import ValidationError

log = get_logger(__name__)

router = APIRouter(prefix="/detect", tags=["detection"])


def _to_response(result: DetectionResult, severity: str | None) -> DetectionResponse:
    return DetectionResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        is_attack=result.is_attack,
        severity=severity,
        timestamp=result.timestamp,
        defaulted_features=result.defaulted_features,
        class_probabilities=result.class_probabilities,
    )


def _severity_for(result: DetectionResult) -> str | None:
    manager = service_state.state.alert_manager
    if manager is None or not result.is_attack:
        return None
    return manager.grade(result.confidence, result.predicted_class).value


@router.post("", response_model=DetectionResponse, summary="Detect intrusion in a single flow")
def detect(flow: FlowInput) -> DetectionResponse:
    """Classify one network flow and raise an alert when it is malicious."""
    engine = service_state.state.require_engine()
    try:
        result = engine.predict_one(flow.to_record())
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    manager = service_state.state.alert_manager
    if manager is not None:
        manager.handle(result)
    return _to_response(result, _severity_for(result))


@router.post("/batch", response_model=BatchDetectionResponse, summary="Detect intrusions in a batch")
def detect_batch(payload: BatchFlowInput) -> BatchDetectionResponse:
    """Classify a batch of flows through the streaming processor."""
    processor = service_state.state.processor
    if processor is None:
        service_state.state.require_engine()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stream processor unavailable"
        )

    try:
        results = processor.process_batch([flow.to_record() for flow in payload.flows])
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    responses = [_to_response(result, _severity_for(result)) for result in results]
    return BatchDetectionResponse(
        count=len(responses),
        attacks=sum(1 for item in responses if item.is_attack),
        results=responses,
    )


@router.get("/stats", response_model=DetectionStats, summary="Recent detection statistics")
def detection_stats(limit: int = Query(20, ge=1, le=200)) -> DetectionStats:
    """Return rolling detection counters and the most recent alerts."""
    processor = service_state.state.processor
    manager = service_state.state.alert_manager
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=service_state.state.load_error or "No detection model is loaded",
        )

    lifetime = processor.lifetime_stats()
    alerts = [AlertOutput(**alert.to_dict()) for alert in manager.recent(limit)] if manager else []
    return DetectionStats(
        total_processed=lifetime["total_processed"],
        total_attacks=lifetime["total_attacks"],
        lifetime_attack_rate=lifetime["lifetime_attack_rate"],
        window=lifetime["window"],
        alert_severity_counts=manager.severity_counts() if manager else {},
        recent_alerts=alerts,
    )
