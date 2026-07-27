"""Pydantic request and response models for the detection API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FlowInput(BaseModel):
    """A single network flow record.

    Known KDD-style fields are typed explicitly; any additional feature the
    trained model expects may be supplied as an extra key.
    """

    model_config = ConfigDict(extra="allow")

    duration: float = Field(0.0, ge=0, description="Flow duration in seconds")
    protocol_type: str = Field("tcp", description="Transport protocol")
    service: str = Field("http", description="Destination service")
    flag: str = Field("SF", description="Connection status flag")
    src_bytes: int = Field(0, ge=0, description="Bytes sent by the source")
    dst_bytes: int = Field(0, ge=0, description="Bytes sent by the destination")
    count: int = Field(1, ge=0, description="Connections to the same host in the last 2 seconds")
    srv_count: int = Field(1, ge=0, description="Connections to the same service in the last 2 seconds")

    @field_validator("protocol_type", "service", "flag")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value.strip()

    def to_record(self) -> dict[str, Any]:
        """Flatten the model, including extra features, into a plain dict."""
        return self.model_dump()


class BatchFlowInput(BaseModel):
    """A batch of flows submitted for detection."""

    flows: list[FlowInput] = Field(..., min_length=1, max_length=5000)


class DetectionResponse(BaseModel):
    """Detection outcome for one flow."""

    predicted_class: str
    confidence: float
    is_attack: bool
    severity: str | None = None
    timestamp: str
    defaulted_features: int = Field(
        0, description="Model inputs absent from the request and filled with training defaults"
    )
    class_probabilities: dict[str, float] = Field(default_factory=dict)


class BatchDetectionResponse(BaseModel):
    """Detection outcomes for a batch, with a short roll-up."""

    count: int
    attacks: int
    results: list[DetectionResponse]


class AlertOutput(BaseModel):
    """An alert as exposed over the API."""

    alert_id: str
    timestamp: str
    severity: str
    predicted_class: str
    confidence: float
    message: str
    flow_summary: dict[str, Any] = Field(default_factory=dict)


class DetectionStats(BaseModel):
    """Rolling detection statistics and alert counters."""

    total_processed: int
    total_attacks: int
    lifetime_attack_rate: float
    window: dict[str, Any]
    alert_severity_counts: dict[str, int]
    recent_alerts: list[AlertOutput] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health and model status."""

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Uniform error payload."""

    detail: str
    error_type: str
