"""Tests for configuration and validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.config import load_config
from src.utils.validators import (
    ValidationError,
    ensure_directory,
    validate_dataframe,
    validate_flow,
    validate_path,
    validate_ratio,
    validate_target,
)


def test_load_config_defaults_when_file_missing(tmp_path) -> None:
    config = load_config(tmp_path / "absent.yaml")
    assert config.app.name == "NIDS-ML"
    assert config.data.test_size == 0.15


def test_load_config_reads_yaml(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("app:\n  name: Custom\napi:\n  port: 9001\n", encoding="utf-8")
    config = load_config(path)
    assert config.app.name == "Custom"
    assert config.api.port == 9001


def test_environment_overrides_config(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("api:\n  port: 8000\n", encoding="utf-8")
    monkeypatch.setenv("NIDS_API__PORT", "9999")
    assert load_config(path).api.port == 9999


def test_config_rejects_non_mapping(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_config(path)


def test_resolve_makes_paths_absolute() -> None:
    assert load_config().resolve("models").is_absolute()


def test_validate_dataframe_rules() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    assert validate_dataframe(frame, required_columns=["a"]) is frame
    with pytest.raises(ValidationError):
        validate_dataframe(frame, required_columns=["missing"])
    with pytest.raises(ValidationError):
        validate_dataframe(pd.DataFrame({"a": []}))
    with pytest.raises(ValidationError):
        validate_dataframe([1, 2, 3])


def test_validate_target_rules() -> None:
    frame = pd.DataFrame({"label": ["normal", "dos"]})
    assert len(validate_target(frame, "label")) == 2
    with pytest.raises(ValidationError):
        validate_target(pd.DataFrame({"label": ["normal", "normal"]}), "label")
    with pytest.raises(ValidationError):
        validate_target(pd.DataFrame({"label": ["normal", None]}), "label")


def test_validate_path_rules(tmp_path) -> None:
    path = tmp_path / "flows.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert validate_path(path, suffixes=[".csv"]) == path
    with pytest.raises(ValidationError):
        validate_path(tmp_path / "absent.csv")
    with pytest.raises(ValidationError):
        validate_path(path, suffixes=[".parquet"])


def test_validate_ratio_rules() -> None:
    assert validate_ratio(0.5, "test_size") == 0.5
    with pytest.raises(ValidationError):
        validate_ratio(1.5, "test_size")
    with pytest.raises(ValidationError):
        validate_ratio("half", "test_size")


def test_validate_flow_rules() -> None:
    flow = {"duration": 1.0, "src_bytes": 100}
    assert validate_flow(flow, ["duration", "src_bytes"]) == flow
    with pytest.raises(ValidationError):
        validate_flow(flow, ["duration", "missing"])
    with pytest.raises(ValidationError):
        validate_flow({"duration": float("inf")}, ["duration"])
    with pytest.raises(ValidationError):
        validate_flow({"duration": None}, ["duration"])


def test_ensure_directory_creates_nested_path(tmp_path) -> None:
    created = ensure_directory(tmp_path / "a" / "b")
    assert created.is_dir()
