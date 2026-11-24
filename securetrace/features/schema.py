from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

IDENTITY_COLUMN_HINTS = {
    "flow_id",
    "source_ip",
    "src_ip",
    "destination_ip",
    "dst_ip",
    "source_port",
    "src_port",
    "destination_port",
    "dst_port",
    "timestamp",
    "time",
    "flow_start_time",
    "flow_end_time",
    "start_time",
    "end_time",
    "session_id",
    "segment_id",
    "source_mac",
    "destination_mac",
    "protocol",
}

# CSE-CIC-IDS2018 CICFlowMeter column names -> the corrected CIC-IDS2017 names.
# The two releases name the same flow statistics differently, so an exact-name
# intersection keeps only ~32 columns; this map recovers the rest. Columns that
# exist in only one release (e.g. "Pkt Len Var", "Pkt Size Avg", "Bwd Act Data
# Pkts") have no entry and fall out of the shared feature set.
IDS2018_TO_CANONICAL = {
    "Tot Fwd Pkts": "Total Fwd Packet",
    "Tot Bwd Pkts": "Total Bwd packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packet",
    "TotLen Bwd Pkts": "Total Length of Bwd Packet",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Packet Length Min",
    "Pkt Len Max": "Packet Length Max",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Count": "CWR Flag Count",  # sic: CWE is a typo for CWR in the 2018 header
    "ECE Flag Cnt": "ECE Flag Count",
    "Fwd Seg Size Avg": "Fwd Segment Size Avg",
    "Bwd Seg Size Avg": "Bwd Segment Size Avg",
    "Fwd Byts/b Avg": "Fwd Bytes/Bulk Avg",
    "Fwd Pkts/b Avg": "Fwd Packet/Bulk Avg",
    "Fwd Blk Rate Avg": "Fwd Bulk Rate Avg",
    "Bwd Byts/b Avg": "Bwd Bytes/Bulk Avg",
    "Bwd Pkts/b Avg": "Bwd Packet/Bulk Avg",
    "Bwd Blk Rate Avg": "Bwd Bulk Rate Avg",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "FWD Init Win Bytes",
    "Init Bwd Win Byts": "Bwd Init Win Bytes",
}


def harmonize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename 2018-style CICFlowMeter columns to the canonical (2017-corrected) names."""
    return frame.rename(columns=IDS2018_TO_CANONICAL)


def _canonical_name(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    return re.sub(r"_+", "_", text)


def _is_identity_column(canon: str) -> bool:
    if canon in IDENTITY_COLUMN_HINTS:
        return True
    if "_ip" in canon or "ip_address" in canon:
        return True
    if canon.endswith("_id"):
        return True
    if "timestamp" in canon:
        return True
    # Wall-clock time columns, but not duration-style statistics.
    return "time" in canon and "duration" not in canon


def drop_identity_features(frame: pd.DataFrame, extra_drop: Iterable[str] | None = None) -> pd.DataFrame:
    extra_drop = set(extra_drop or [])
    drop_columns = {str(column) for column in frame.columns if _is_identity_column(_canonical_name(column))}
    drop_columns |= {column for column in extra_drop if column in frame.columns}
    return frame.drop(columns=list(drop_columns), errors="ignore")


def prepare_features(
    frame: pd.DataFrame, label_col: str = "Label", *, drop_identity: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop identity columns, coerce to numeric and replace inf with NaN.

    Row order and index are preserved (no row is dropped here) so that day and
    dataset metadata stay aligned. De-duplication is done by the loader, which
    knows about days and datasets.
    """
    working = frame.copy() if not drop_identity else frame
    if drop_identity:
        working = drop_identity_features(working)

    if label_col not in working.columns:
        raise ValueError(f"Label column '{label_col}' not found in the dataframe.")
    labels = working.pop(label_col)

    for column in working.columns:
        if not pd.api.types.is_numeric_dtype(working[column]):
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan)
    working = working.dropna(axis=1, how="all")
    return working, labels


BENIGN_LABELS = {"benign", "background", "normal"}


def is_attempted_label(labels: pd.Series) -> pd.Series:
    """True for '... - Attempted' rows (corrected CIC-IDS2017): attack tooling ran but
    the flow carried no real attack, so it is neither a clean positive nor a clean negative."""
    return labels.astype(str).str.lower().str.contains("attempted", regex=False)


def label_to_binary(labels: pd.Series) -> pd.Series:
    """0 for benign/background/normal, 1 for any other (attack-family) label.

    Missing labels raise instead of silently becoming attacks.
    """
    if labels.isna().any():
        raise ValueError("Missing labels cannot be mapped to binary; drop them first.")
    normalized = labels.astype(str).str.strip().str.lower()
    return (~normalized.isin(BENIGN_LABELS)).astype(int)
