"""Unsupervised domain adaptation, public (labelled source) -> lab (unlabelled target): CORAL and DANN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from torch import nn

from securetrace.unsup.prep import make_prep


def _fit_prep(X_src: pd.DataFrame, X_tgt: pd.DataFrame):
    prep = make_prep().fit(pd.concat([X_src, X_tgt], ignore_index=True))
    return prep


def coral_scores(X_src, y_src, X_tgt_train, X_tgt_test, seed: int) -> np.ndarray:
    """CORAL (Sun et al. 2016): re-colour the source features so their covariance matches the
    target's, then train a plain classifier on the aligned source."""
    prep = _fit_prep(X_src, X_tgt_train)
    S, T, T_test = prep.transform(X_src), prep.transform(X_tgt_train), prep.transform(X_tgt_test)
    eye = np.eye(S.shape[1])
    cs, ct = np.cov(S, rowvar=False) + eye, np.cov(T, rowvar=False) + eye
    whiten = linalg.fractional_matrix_power(cs, -0.5)
    recolour = linalg.fractional_matrix_power(ct, 0.5)
    S_aligned = np.real(S @ whiten @ recolour)
    clf = LogisticRegression(C=0.1, max_iter=500, random_state=seed).fit(S_aligned, y_src)
    return clf.predict_proba(T_test)[:, 1]


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def dann_scores(X_src, y_src, X_tgt_train, X_tgt_test, seed: int, epochs: int = 40, batch: int = 256) -> np.ndarray:
    """DANN (Ganin et al. 2016): a shared feature extractor trained to classify the labelled source
    while fooling a domain discriminator that tries to tell source from lab-target flows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prep = _fit_prep(X_src, X_tgt_train)
    S = torch.tensor(prep.transform(X_src), dtype=torch.float32)
    T = torch.tensor(prep.transform(X_tgt_train), dtype=torch.float32)
    T_test = torch.tensor(prep.transform(X_tgt_test), dtype=torch.float32)
    ys = torch.tensor(y_src, dtype=torch.float32)
    d = S.shape[1]
    feat = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
    cls = nn.Linear(64, 1)
    dom = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(list(feat.parameters()) + list(cls.parameters()) + list(dom.parameters()), lr=1e-3)
    bce = nn.functional.binary_cross_entropy_with_logits
    steps_per_epoch = max(1, len(S) // batch)
    total, step = epochs * steps_per_epoch, 0
    for _ in range(epochs):
        for _s in range(steps_per_epoch):
            lam = 2.0 / (1.0 + np.exp(-10.0 * step / total)) - 1.0  # standard DANN ramp-up
            si = rng.choice(len(S), size=batch, replace=False)
            ti = rng.choice(len(T), size=min(batch, len(T)), replace=len(T) < batch)
            fs, ft = feat(S[si]), feat(T[ti])
            loss_cls = bce(cls(fs).squeeze(-1), ys[si])
            z = torch.cat([_GradReverse.apply(fs, lam), _GradReverse.apply(ft, lam)])
            dlabel = torch.cat([torch.zeros(len(fs)), torch.ones(len(ft))])
            loss = loss_cls + bce(dom(z).squeeze(-1), dlabel)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    feat.eval()
    with torch.no_grad():
        return torch.sigmoid(cls(feat(T_test)).squeeze(-1)).numpy()
