from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from securetrace.features.schema import (
    IDS2018_TO_CANONICAL,
    drop_identity_features,
    harmonize_columns,
    is_attempted_label,
    label_to_binary,
    prepare_features,
)

# Columns that identify a flow or a host. They are removed from the model input
# (prepare_features also drops anything that looks like one of these).
RAW_IDENTITY_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"}


def load_csv_dataset(csv_path: str | Path, *, label_col: str = "Label") -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    if "Unnamed: 0" in frame.columns:
        frame = frame.drop(columns=["Unnamed: 0"])
    return frame


def _detect_dataset(columns: Iterable[str], path: str | Path) -> str:
    if any(column in IDS2018_TO_CANONICAL for column in columns):
        return "ids2018"
    name = Path(path).name.lower()
    if "insdn" in name:
        return "insdn"
    return "ids2018" if "2018" in name and "2017" not in name else "ids2017"


def _parse_days(timestamps: pd.Series) -> pd.Series:
    """Calendar day as 'YYYY-MM-DD'. 2017 is ISO, 2018 is dd/mm/YYYY; invalid -> NaT."""
    parsed = pd.to_datetime(timestamps, format="ISO8601", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    # A handful of 2018 rows carry epoch-era garbage timestamps (1970); treat as invalid.
    parsed = parsed.where(parsed.dt.year >= 2000)
    return parsed.dt.strftime("%Y-%m-%d")


def _clean_chunk(frame, shared, label_col, day_col, dataset, path, index, drop_attempted, deduplicate):
    """Validity filtering, day parsing, float32 features and (chunk-local) dedup for one chunk."""
    # Stray repeated header rows and unlabeled rows are corrupt, not attacks.
    valid = frame[label_col].notna() & (frame[label_col].astype(str).str.strip() != label_col)

    if day_col in frame.columns:
        days = frame[day_col].astype(str)
    elif "Timestamp" in frame.columns:
        days = _parse_days(frame["Timestamp"])
        valid &= days.notna()
    else:
        digits = "".join(ch for ch in str(path) if ch.isdigit())
        days = pd.Series([digits[:8] or str(index + 1)] * len(frame), index=frame.index)

    if drop_attempted:
        valid &= ~is_attempted_label(frame[label_col])

    columns = sorted(c for c in frame.columns if c in shared)
    # Time-like statistics (e.g. "Total Connection Flow Time") are dropped before dedup.
    columns = [c for c in columns if c in drop_identity_features(frame[[c]]).columns]
    kept = frame.loc[valid, columns + [label_col]]
    features = pd.DataFrame(
        {c: pd.to_numeric(kept[c], errors="coerce").astype(np.float32) for c in columns}, index=kept.index
    )
    features[label_col] = kept[label_col]
    meta = pd.DataFrame(
        {"dataset": dataset, "day": days[valid], "label": kept[label_col].astype(str).str.strip()},
        index=kept.index,
    )
    if deduplicate:
        keep = ~features.assign(_label=meta["label"]).duplicated(keep="first")
        features, meta = features[keep], meta[keep]
    return features, meta


def load_public_frame(
    csv_paths: Sequence[str | Path],
    *,
    label_col: str = "Label",
    day_col: str = "Day",
    drop_attempted: bool = True,
    deduplicate: bool = True,
    drop_features: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load public CICFlowMeter CSVs into one aligned feature matrix.

    Returns (X, y, meta): X float32 features on the columns shared by every file
    (after harmonising names), y binary (1 = attack), meta with `dataset`, `day`
    and the raw `label`. Identity columns and timestamps never reach X.
    """
    # Pass 1: headers only, to find the columns shared by every file (after harmonising names).
    headers = [pd.read_csv(path, nrows=0).columns.tolist() for path in csv_paths]
    datasets = [_detect_dataset(header, path) for header, path in zip(headers, csv_paths)]
    shared = set.intersection(*({IDS2018_TO_CANONICAL.get(c, c) for c in header} for header in headers))
    shared -= {label_col, "Unnamed: 0"} | RAW_IDENTITY_COLUMNS | set(drop_features)
    wanted = shared | {label_col, day_col, "Timestamp"}

    # Pass 2: read only those columns, in chunks, as float32, to keep memory bounded
    # (InSDN alone is ~6M rows).
    cleaned_frames: list[pd.DataFrame] = []
    meta_frames: list[pd.DataFrame] = []
    for index, (path, dataset) in enumerate(zip(csv_paths, datasets)):
        file_features: list[pd.DataFrame] = []
        file_meta: list[pd.DataFrame] = []
        reader = pd.read_csv(
            path,
            low_memory=False,
            chunksize=500_000,
            usecols=lambda c: IDS2018_TO_CANONICAL.get(c, c) in wanted,
        )
        for chunk in reader:
            features, meta = _clean_chunk(
                harmonize_columns(chunk), shared, label_col, day_col, dataset, path, index, drop_attempted, deduplicate
            )
            file_features.append(features)
            file_meta.append(meta)
        features = pd.concat(file_features, ignore_index=True)
        meta = pd.concat(file_meta, ignore_index=True)
        if deduplicate:
            keep = ~features.assign(_label=meta["label"]).duplicated(keep="first")
            features, meta = features[keep], meta[keep]
        cleaned_frames.append(features)
        meta_frames.append(meta)

    combined = pd.concat(cleaned_frames, ignore_index=True)
    meta = pd.concat(meta_frames, ignore_index=True)

    X, _ = prepare_features(combined, label_col=label_col)
    y = label_to_binary(meta["label"])

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    return X, y, meta


def load_and_clean_public_data(
    csv_paths: Sequence[str | Path],
    *,
    label_col: str = "Label",
    day_col: str = "Day",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, y, meta = load_public_frame(csv_paths, label_col=label_col, day_col=day_col)
    return X, y, meta["day"]


def split_by_day(
    features: pd.DataFrame,
    labels: pd.Series,
    day_labels: pd.Series,
    *,
    train_days: Iterable | None = None,
    val_days: Iterable | None = None,
    test_days: Iterable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split by calendar day, never by row. A day may belong to at most one split."""
    days = day_labels.astype(str)
    unique_days = sorted(days.dropna().unique())
    if not unique_days:
        raise ValueError("No day labels were found for a day-wise split.")

    def as_set(values: Iterable | None) -> set[str] | None:
        return None if values is None else {str(v) for v in values}

    train_set, val_set, test_set = as_set(train_days), as_set(val_days), as_set(test_days)
    if train_set is None:
        train_set = set(unique_days[:-2])
    if val_set is None:
        val_set = {unique_days[-2]} if len(unique_days) >= 2 else set()
    if test_set is None:
        test_set = {unique_days[-1]}

    for name_a, a, name_b, b in (
        ("train", train_set, "val", val_set),
        ("train", train_set, "test", test_set),
        ("val", val_set, "test", test_set),
    ):
        overlap = a & b
        if overlap:
            raise ValueError(f"Day(s) {sorted(overlap)} appear in both {name_a} and {name_b}.")

    train_mask = days.isin(train_set)
    val_mask = days.isin(val_set)
    test_mask = days.isin(test_set)

    return (
        features.loc[train_mask],
        features.loc[val_mask],
        features.loc[test_mask],
        labels.loc[train_mask],
        labels.loc[val_mask],
        labels.loc[test_mask],
    )
