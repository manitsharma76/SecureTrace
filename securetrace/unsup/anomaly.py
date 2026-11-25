"""Anomaly-detection comparators. Both are fit on UNLABELLED lab flows (train sessions, which are
contaminated with attacks: nothing in a real deployment says which flows are clean) and score the
lab test flows; a higher score means 'more anomalous'."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from securetrace.unsup.prep import make_prep


def isolation_forest_scores(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int) -> np.ndarray:
    prep = make_prep().fit(X_train)
    forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(prep.transform(X_train))
    return -forest.score_samples(prep.transform(X_test))


def autoencoder_scores(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int, epochs: int = 60) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prep = make_prep().fit(X_train)
    Xt = torch.tensor(prep.transform(X_train), dtype=torch.float32)
    d = Xt.shape[1]
    net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
                        nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = rng.permutation(len(Xt))
        for i in range(0, len(order), 128):
            batch = Xt[order[i:i + 128]]
            loss = nn.functional.mse_loss(net(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        Xe = torch.tensor(prep.transform(X_test), dtype=torch.float32)
        return ((net(Xe) - Xe) ** 2).mean(dim=1).numpy()
