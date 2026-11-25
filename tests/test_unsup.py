import numpy as np
import pandas as pd
import torch

from securetrace.unsup.anomaly import autoencoder_scores, isolation_forest_scores
from securetrace.unsup.da import coral_scores, dann_scores
from securetrace.unsup.prep import shared_columns
from securetrace.unsup.ssl import _corrupt, embed, info_nce, make_encoder, scarf_pretrain


def _frame(n, d=6, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)) + shift, columns=[f"f{i}" for i in range(d)])


def test_anomaly_detectors_rank_obvious_outliers_highest():
    train = _frame(400)
    test = pd.concat([_frame(50, seed=1), _frame(10, seed=2, shift=8.0)], ignore_index=True)
    is_outlier = np.r_[np.zeros(50), np.ones(10)]
    for scores in (isolation_forest_scores(train, test, 0), autoencoder_scores(train, test, 0, epochs=30)):
        assert scores[is_outlier == 1].mean() > scores[is_outlier == 0].mean()


def test_info_nce_prefers_matched_pairs():
    z = torch.randn(32, 16)
    assert info_nce(z, z + 0.01 * torch.randn_like(z)).item() < info_nce(z, torch.randn(32, 16)).item()


def test_scarf_corruption_keeps_shape_and_only_uses_batch_values():
    gen = torch.Generator().manual_seed(0)
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    out = _corrupt(x, 0.5, gen)
    assert out.shape == x.shape
    for j in range(4):
        assert set(out[:, j].tolist()) <= set(x[:, j].tolist())  # donors come from the same feature column
    assert not torch.equal(out, x)


def test_scarf_pretraining_yields_embeddings_of_expected_size():
    X = np.random.default_rng(0).normal(size=(300, 6)).astype(np.float32)
    enc = scarf_pretrain(X, seed=0, epochs=3, batch=64)
    assert embed(enc, X).shape == (300, 64)
    assert embed(make_encoder(6), X).shape == (300, 64)


def test_domain_adaptation_methods_return_one_score_per_target_flow():
    src = _frame(300)
    y = (src["f0"] > 0.5).astype(int).to_numpy()
    tgt_train, tgt_test = _frame(200, seed=3, shift=0.5), _frame(80, seed=4, shift=0.5)
    for fn in (coral_scores, lambda *a: dann_scores(*a, epochs=3)):
        scores = fn(src, y, tgt_train, tgt_test, 0)
        assert scores.shape == (80,) and np.isfinite(scores).all()


def test_coral_still_classifies_when_source_has_signal():
    src = _frame(800)
    y = (src["f0"] > 0).astype(int).to_numpy()
    tgt_test = _frame(400, seed=5)
    truth = (tgt_test["f0"] > 0).astype(int).to_numpy()
    scores = coral_scores(src, y, _frame(400, seed=6), tgt_test, 0)
    assert scores[truth == 1].mean() > scores[truth == 0].mean()


def test_shared_columns_drops_columns_the_lab_never_populates():
    lab = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    assert shared_columns(["a", "b", "c"], lab) == ["a"]
