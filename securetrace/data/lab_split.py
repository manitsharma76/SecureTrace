"""Session-level train/val/test split for the lab dataset, with one attack type (mutator) held
out of train/val entirely -- CLAUDE.md "One attack type stays held out of training and is used
only for testing" and DEVELOPMENT_PLAN.md "split by session into train / validation / test, with
one attack type held out."

Splitting happens on whole SESSIONS (never on individual flow rows, per CLAUDE.md "split by
session"); every flow from a given run_id ends up in exactly one of train/val/test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from securetrace.data.lab import load_session_manifest


@dataclass(frozen=True)
class SessionSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]
    held_out_mutator: str | None


def split_lab_sessions(
    manifest_paths: list[Path],
    *,
    held_out_mutator: str,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> SessionSplit:
    """`held_out_mutator` sessions all go to test, regardless of `test_fraction`; the remaining
    (clean + other-mutator) sessions are split train/val/test by the given fractions, each class
    (clean / each other mutator) shuffled and split separately so train and val both see every
    non-held-out class, and no session is split across sets."""
    rng = random.Random(seed)

    mutator_by_path = {path: load_session_manifest(path)["plan"]["mutator"] for path in manifest_paths}

    held_out = [p for p in manifest_paths if mutator_by_path[p] == held_out_mutator]
    remaining = [p for p in manifest_paths if mutator_by_path[p] != held_out_mutator]

    by_class: dict[str | None, list[Path]] = {}
    for path in remaining:
        by_class.setdefault(mutator_by_path[path], []).append(path)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = list(held_out)
    for paths in by_class.values():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = round(n * test_fraction)
        n_val = round(n * val_fraction)
        test += shuffled[:n_test]
        val += shuffled[n_test : n_test + n_val]
        train += shuffled[n_test + n_val :]

    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise AssertionError(f"session(s) ended up in two splits: {overlap}")  # should be unreachable

    return SessionSplit(train=train, val=val, test=test, held_out_mutator=held_out_mutator)
