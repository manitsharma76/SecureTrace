import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from securetrace.adapt.data import subsample_tamper_sessions
from securetrace.adapt.neural import from_public_pipeline, score_neural, fit_neural
from securetrace.adapt.pu import (
    elkan_noto_dataset, estimate_class_prior, estimate_label_frequency, fit_labelled_vs_unlabelled, nnpu_loss,
)
from securetrace.adapt.tree import adapt_xgboost
from securetrace.models.factory import build_model


def _synthetic(n=1200, seed=0, c=0.5):
    """y ~ Bernoulli(0.2); feature 0 separates y; s labels a random fraction c of the positives (SCAR)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    X["a"] += 3.0 * y
    s = ((y == 1) & (rng.random(n) < c)).astype(int)
    return X, y, s


def test_elkan_noto_estimate_is_a_biased_low_bound_under_scar():
    """e1 underestimates c when positives overlap other flows in feature space (here ~0.36 vs a true
    0.5), so the prior pi = P(s=1)/c overshoots (~0.28 vs 0.2). The test pins that known direction
    of bias instead of pretending the estimator is exact."""
    X, y, s = _synthetic(n=6000, c=0.5)
    half = len(X) // 2
    g = fit_labelled_vs_unlabelled(X.iloc[:half], s[:half], 0)
    c_hat = estimate_label_frequency(g, X.iloc[half:], s[half:])
    assert 0.25 < c_hat < 0.5
    prior = estimate_class_prior(s, c_hat)
    assert 0.2 <= prior < 0.45


def test_elkan_noto_weights_conserve_each_unlabelled_flow():
    X, y, s = _synthetic(n=300)
    g = np.full(len(s), 0.3)
    X_all, y_all, w_all = elkan_noto_dataset(X, s, g, c=0.5)
    n_unlabelled = int((s == 0).sum())
    # every unlabelled flow contributes total weight 1 (positive copy w + negative copy 1-w);
    # labelled positives contribute 1 each.
    assert w_all.sum() == pytest.approx(n_unlabelled + int(s.sum()))
    assert set(np.unique(y_all)) <= {0.0, 1.0}


def test_nnpu_loss_prefers_correct_scores_and_is_nonnegative_branch_safe():
    z_p_good, z_u_good = torch.full((50,), 4.0), torch.cat([torch.full((40,), -4.0), torch.full((10,), 4.0)])
    z_p_bad, z_u_bad = torch.full((50,), -4.0), torch.full((50,), 4.0)
    good = nnpu_loss(z_p_good, z_u_good, prior=0.2)
    bad = nnpu_loss(z_p_bad, z_u_bad, prior=0.2)
    assert good.item() < bad.item()
    # All-unlabelled scored as strongly positive with P also positive: the negative risk goes
    # negative and the loss switches to the corrective (ascent) objective instead of exploding.
    out = nnpu_loss(torch.full((50,), 4.0), torch.full((50,), 4.0), prior=0.9)
    assert torch.isfinite(out)


def test_public_mlp_weights_transfer_exactly_to_torch():
    X, y, _ = _synthetic(n=600)
    pipe = build_model("mlp", {"hidden": [16, 8], "max_iter": 30}, seed=0).fit(X, y)
    preprocess, net = from_public_pipeline(pipe)
    ours = score_neural(preprocess, net, X)
    assert np.allclose(ours, pipe.predict_proba(X)[:, 1], atol=1e-4)


def test_adaptation_gain_vanishes_with_shuffled_weak_labels():
    """The Novelty A control: real weak labels lift a model that starts at chance; the same method
    on shuffled labels must not."""
    X, y, s = _synthetic(n=2400, c=0.8, seed=1)
    X_test, y_test, _ = _synthetic(n=1500, seed=2)
    groups = np.repeat(np.arange(12), len(X) // 12)
    # A "public" model that never saw the informative feature (a is constant for it): chance level.
    public = XGBClassifier(n_estimators=5, max_depth=1, random_state=0).fit(X.assign(a=0.0), y)
    before = average_precision_score(y_test, public.predict_proba(X_test.assign(a=0.0))[:, 1])

    real, _ = adapt_xgboost(public, X, s, groups, X, s, mode="naive", seed=0, n_new_trees=30)
    real_ap = average_precision_score(y_test, real.predict_proba(X_test)[:, 1])
    s_shuf = np.random.default_rng(3).permutation(s)
    shuf, _ = adapt_xgboost(public, X, s_shuf, groups, X, s_shuf, mode="naive", seed=0, n_new_trees=30)
    shuf_ap = average_precision_score(y_test, shuf.predict_proba(X_test)[:, 1])

    assert real_ap > before + 0.3
    assert shuf_ap < real_ap - 0.3
    assert shuf_ap < 0.45  # near the 0.2 attack rate


def test_fit_neural_rejects_unknown_mode():
    X, y, s = _synthetic(n=200)
    pipe = build_model("mlp", {"hidden": [8], "max_iter": 20}, seed=0).fit(X, y)
    preprocess, net = from_public_pipeline(pipe)
    with pytest.raises(ValueError):
        fit_neural(preprocess, net, X, s, np.array(["b"] * len(s)), mode="bogus", prior=0.2, seed=0)


def test_subsample_tamper_sessions_keeps_all_clean(tmp_path):
    paths = []
    for i, seg in enumerate([None, None, "A", "B", "C", "A", "B", "C"]):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps({"run_id": f"r{i}", "plan": {"tamper_segment": seg, "mutator": None}}))
        paths.append(p)
    kept = subsample_tamper_sessions(paths, 0.5, seed=0)
    assert {paths[0], paths[1]} <= set(kept)
    assert len(kept) == 2 + 3
    assert subsample_tamper_sessions(paths, 1.0, seed=0) == paths
