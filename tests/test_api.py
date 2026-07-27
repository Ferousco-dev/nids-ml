"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.alerting.alert_manager import AlertManager
from src.alerting.notifier import Notifier
from src.alerting.throttler import AlertThrottler
from src.api import state as service_state
from src.api.main import create_app
from src.detection.stream_processor import StreamProcessor


class SilentChannel:
    name = "silent"

    def send(self, alert) -> None:
        return None


@pytest.fixture()
def client(engine, monkeypatch) -> TestClient:
    """A test client whose service state is wired to the fixture engine."""
    manager = AlertManager(
        notifier=Notifier([SilentChannel()]), throttler=AlertThrottler(window_seconds=60, max_alerts=50)
    )
    processor = StreamProcessor(engine, batch_size=25, on_detection=manager.handle)

    def fake_initialise(config=None):
        service_state.state.engine = engine
        service_state.state.processor = processor
        service_state.state.alert_manager = manager
        service_state.state.load_error = None
        return service_state.state

    monkeypatch.setattr(service_state, "initialise", fake_initialise)
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def flow_payload(raw_dataset: pd.DataFrame) -> dict:
    return raw_dataset.drop(columns=["label"]).iloc[0].to_dict()


def test_root_banner(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_reports_loaded_model(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_name"] == "random_forest"


def test_metrics_endpoint_is_prometheus_text(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "nids_model_loaded 1" in response.text
    assert "nids_flows_processed_total" in response.text


def test_detect_single_flow(client: TestClient, flow_payload: dict) -> None:
    response = client.post("/detect", json=flow_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["is_attack"], bool)


def test_detect_rejects_invalid_payload(client: TestClient, flow_payload: dict) -> None:
    invalid = {**flow_payload, "src_bytes": -10}
    assert client.post("/detect", json=invalid).status_code == 422


def test_detect_accepts_partial_flow(client: TestClient) -> None:
    response = client.post("/detect", json={"duration": 1.0, "count": 300})
    assert response.status_code == 200
    assert response.json()["predicted_class"]


def test_detect_batch(client: TestClient, raw_dataset: pd.DataFrame) -> None:
    flows = raw_dataset.drop(columns=["label"]).head(15).to_dict(orient="records")
    response = client.post("/detect/batch", json={"flows": flows})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 15
    assert len(body["results"]) == 15
    assert body["attacks"] <= 15


def test_detect_batch_rejects_empty_list(client: TestClient) -> None:
    assert client.post("/detect/batch", json={"flows": []}).status_code == 422


def test_detection_stats_after_traffic(client: TestClient, raw_dataset: pd.DataFrame) -> None:
    flows = raw_dataset.drop(columns=["label"]).head(30).to_dict(orient="records")
    client.post("/detect/batch", json={"flows": flows})

    body = client.get("/detect/stats", params={"limit": 5}).json()
    assert body["total_processed"] >= 30
    assert 0.0 <= body["lifetime_attack_rate"] <= 1.0
    assert len(body["recent_alerts"]) <= 5
    assert set(body["alert_severity_counts"]) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_endpoints_degrade_without_model(monkeypatch, flow_payload: dict) -> None:
    def no_model(config=None):
        service_state.state.engine = None
        service_state.state.processor = None
        service_state.state.alert_manager = None
        service_state.state.load_error = "No detection model is loaded"
        return service_state.state

    monkeypatch.setattr(service_state, "initialise", no_model)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/detect", json=flow_payload).status_code == 422
        assert client.get("/detect/stats").status_code == 503
