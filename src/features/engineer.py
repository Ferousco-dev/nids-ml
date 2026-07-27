"""Derived features computed from raw network flow records.

All transformations are vectorised and safe on partial inputs: a derived
feature is only added when its source columns are present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.validators import validate_dataframe

log = get_logger(__name__)

EPSILON = 1e-6

DURATION_BINS = [-np.inf, 1.0, 10.0, 60.0, 300.0, np.inf]
DURATION_LABELS = ["instant", "short", "medium", "long", "persistent"]


def _has(frame: pd.DataFrame, *columns: str) -> bool:
    return all(column in frame.columns for column in columns)


def add_traffic_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add byte/packet rate features normalised by flow duration."""
    result = frame.copy()
    if _has(result, "duration", "src_bytes"):
        result["src_byte_rate"] = result["src_bytes"] / (result["duration"] + EPSILON)
    if _has(result, "duration", "dst_bytes"):
        result["dst_byte_rate"] = result["dst_bytes"] / (result["duration"] + EPSILON)
    if _has(result, "src_bytes", "dst_bytes"):
        result["total_bytes"] = result["src_bytes"] + result["dst_bytes"]
        result["byte_ratio"] = result["src_bytes"] / (result["dst_bytes"] + EPSILON)
        result["is_upload_heavy"] = (result["byte_ratio"] > 1.0).astype(np.int8)
    if _has(result, "duration", "count"):
        result["packet_rate"] = result["count"] / (result["duration"] + EPSILON)
    if _has(result, "duration", "srv_count"):
        result["srv_packet_rate"] = result["srv_count"] / (result["duration"] + EPSILON)
    return result


def add_connection_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ratios describing how connections spread across hosts and services."""
    result = frame.copy()
    if _has(result, "srv_count", "count"):
        result["srv_count_ratio"] = result["srv_count"] / (result["count"] + EPSILON)
    if _has(result, "dst_host_srv_count", "dst_host_count"):
        result["dst_host_srv_ratio"] = result["dst_host_srv_count"] / (result["dst_host_count"] + EPSILON)
    if _has(result, "serror_rate", "rerror_rate"):
        result["error_rate_total"] = result["serror_rate"] + result["rerror_rate"]
    if _has(result, "dst_host_serror_rate", "dst_host_rerror_rate"):
        result["dst_host_error_rate_total"] = (
            result["dst_host_serror_rate"] + result["dst_host_rerror_rate"]
        )
    if _has(result, "same_srv_rate", "diff_srv_rate"):
        result["srv_diversity"] = result["diff_srv_rate"] - result["same_srv_rate"]
    return result


def add_duration_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket flow duration into ordinal bands."""
    result = frame.copy()
    if "duration" in result.columns:
        result["duration_band"] = pd.cut(
            result["duration"], bins=DURATION_BINS, labels=DURATION_LABELS
        ).astype("object")
        result["log_duration"] = np.log1p(result["duration"].clip(lower=0))
    return result


def add_security_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add composite indicators of privilege escalation and credential abuse."""
    result = frame.copy()
    privilege_columns = [
        column
        for column in ("root_shell", "su_attempted", "num_root", "num_shells", "num_compromised")
        if column in result.columns
    ]
    if privilege_columns:
        result["privilege_activity"] = result[privilege_columns].sum(axis=1)
        result["has_privilege_activity"] = (result["privilege_activity"] > 0).astype(np.int8)
    if _has(result, "num_failed_logins"):
        result["failed_login_flag"] = (result["num_failed_logins"] > 0).astype(np.int8)
    if _has(result, "logged_in", "num_failed_logins"):
        result["suspicious_auth"] = (
            (result["num_failed_logins"] > 0) & (result["logged_in"] == 0)
        ).astype(np.int8)
    if _has(result, "num_file_creations", "num_access_files"):
        result["file_activity"] = result["num_file_creations"] + result["num_access_files"]
    return result


def add_interarrival_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Approximate inter-arrival timing statistics from flow counters."""
    result = frame.copy()
    if _has(result, "duration", "count"):
        result["mean_interarrival"] = result["duration"] / (result["count"] + EPSILON)
    if _has(result, "duration", "srv_count"):
        result["mean_srv_interarrival"] = result["duration"] / (result["srv_count"] + EPSILON)
    if _has(result, "mean_interarrival", "mean_srv_interarrival"):
        result["interarrival_gap"] = (
            result["mean_srv_interarrival"] - result["mean_interarrival"]
        ).abs()
    return result


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering chain to a flow frame."""
    validate_dataframe(frame)
    result = frame
    for step in (
        add_traffic_rates,
        add_connection_ratios,
        add_duration_bins,
        add_security_indicators,
        add_interarrival_stats,
    ):
        result = step(result)

    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    added = len(result.columns) - len(frame.columns)
    log.info("Feature engineering added {} column(s); total {}", added, len(result.columns))
    return result
