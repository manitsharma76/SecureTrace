"""SCARF-style contrastive pretraining of a tabular encoder on unlabelled lab flows (Bahri et al.,
2022): a view is made by replacing a random subset of a row's features with values drawn from the
same feature of other rows in the batch; InfoNCE pulls a row and its corrupted view together."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

EMB = 64


def make_encoder(d_in: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, EMB))


def _corrupt(x: torch.Tensor, rate: float, gen: torch.Generator) -> torch.Tensor:
    mask = torch.rand(x.shape, generator=gen) < rate
    donors = torch.stack([x[torch.randint(len(x), (len(x),), generator=gen), j] for j in range(x.shape[1])], dim=1)
    return torch.where(mask, donors, x)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    a, b = nn.functional.normalize(a, dim=1), nn.functional.normalize(b, dim=1)
    logits = a @ b.T / temperature
    return nn.functional.cross_entropy(logits, torch.arange(len(a)))


def scarf_pretrain(X: np.ndarray, seed: int, epochs: int = 100, batch: int = 256, corruption: float = 0.4,
                   lr: float = 1e-3) -> nn.Sequential:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    encoder = make_encoder(Xt.shape[1])
    proj = nn.Sequential(nn.ReLU(), nn.Linear(EMB, EMB))
    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=lr)
    for _ in range(epochs):
        order = rng.permutation(len(Xt))
        for i in range(0, len(order) - batch + 1, batch):
            x = Xt[order[i:i + batch]]
            loss = info_nce(proj(encoder(x)), proj(encoder(_corrupt(x, corruption, gen))))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return encoder


def embed(encoder: nn.Sequential, X: np.ndarray) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        return encoder(torch.tensor(X, dtype=torch.float32)).numpy()
