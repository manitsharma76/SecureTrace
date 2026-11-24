import pandas as pd

from scripts.evaluate_lab import align_to_training_columns


def test_align_reindexes_to_training_columns_in_order():
    X = pd.DataFrame({"b": [1, 2], "a": [3, 4], "extra": [5, 6]})
    aligned, missing, extra = align_to_training_columns(X, ["a", "b", "c"])
    assert list(aligned.columns) == ["a", "b", "c"]
    assert missing == ["c"]
    assert extra == ["extra"]
    assert aligned["c"].isna().all()
    assert aligned["a"].tolist() == [3, 4]


def test_align_with_identical_columns_has_no_missing_or_extra():
    X = pd.DataFrame({"a": [1], "b": [2]})
    aligned, missing, extra = align_to_training_columns(X, ["a", "b"])
    assert missing == []
    assert extra == []
    assert list(aligned.columns) == ["a", "b"]
