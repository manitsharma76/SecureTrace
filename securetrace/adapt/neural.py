"""Neural adaptation of the public MLP: naive BCE, nnPU on flows, and attention-pooling MIL with a
bag-level nnPU risk. All three start from the public model's weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from securetrace.adapt.pu import nnpu_loss

NEURAL_MODES = ("naive", "nnpu", "mil")


class FlowNet(nn.Module):
    """Encoder (the public MLP's hidden layers) + a linear instance-logit head, plus an attention
    pool used only by MIL. The instance logit z_i is the per-flow score at read-out time."""

    def __init__(self, encoder: nn.Sequential, head: nn.Linear, hidden: int, attn_dim: int = 32):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.attn_v = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)

    def instance_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)

    def bag_logit(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.head(h).squeeze(-1)
        a = torch.softmax(self.attn_w(torch.tanh(self.attn_v(h))).squeeze(-1), dim=0)
        return (a * z).sum()


def from_public_pipeline(pipeline) -> tuple[object, FlowNet]:
    """Split the saved sklearn pipeline into (feature preprocessing, torch net initialised from the
    fitted MLPClassifier weights)."""
    preprocess, mlp = pipeline[:-1], pipeline[-1]
    layers: list[nn.Module] = []
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        lin = nn.Linear(W.shape[0], W.shape[1])
        lin.weight.data = torch.tensor(W.T, dtype=torch.float32)
        lin.bias.data = torch.tensor(b, dtype=torch.float32)
        layers += [lin, nn.ReLU()]
    W, b = mlp.coefs_[-1], mlp.intercepts_[-1]
    head = nn.Linear(W.shape[0], 1)
    head.weight.data = torch.tensor(W.T, dtype=torch.float32)
    head.bias.data = torch.tensor(b, dtype=torch.float32)
    return preprocess, FlowNet(nn.Sequential(*layers), head, hidden=W.shape[0])


def _tensor(preprocess, X: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(np.asarray(preprocess.transform(X), dtype=np.float32))


def fit_neural(
    preprocess, net: FlowNet, X: pd.DataFrame, s: np.ndarray, bag_id: np.ndarray, *, mode: str,
    prior: float, seed: int, epochs: int = 40, lr: float = 3e-4, max_bag: int = 64,
) -> FlowNet:
    if mode not in NEURAL_MODES:
        raise ValueError(f"unknown neural adaptation mode {mode!r}; choose from {NEURAL_MODES}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = _tensor(preprocess, X)
    st = torch.tensor(s, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    net.train()

    if mode == "mil":
        members: dict[str, list[int]] = {}
        for i, b in enumerate(bag_id):
            members.setdefault(b, []).append(i)
        bags = {b: np.asarray(ix) for b, ix in members.items()}
        pos = [b for b, ix in bags.items() if s[ix].any()]
        pos_set = set(pos)
        unl = [b for b in bags if b not in pos_set]
        if not pos or not unl:
            return net

        def logits(names):
            out = []
            for name in names:
                ix = bags[name]
                if len(ix) > max_bag:
                    ix = rng.choice(ix, size=max_bag, replace=False)
                out.append(net.bag_logit(Xt[ix]))
            return torch.stack(out)

        for _ in range(epochs):
            for _step in range(max(1, len(pos) // 8)):
                pb = rng.choice(len(pos), size=min(8, len(pos)), replace=False)
                ub = rng.choice(len(unl), size=min(16, len(unl)), replace=False)
                loss = nnpu_loss(logits([pos[i] for i in pb]), logits([unl[i] for i in ub]), prior)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return net

    pos_ix, unl_ix = np.flatnonzero(s == 1), np.flatnonzero(s == 0)
    if len(pos_ix) == 0 or len(unl_ix) == 0:
        return net
    steps = max(1, len(Xt) // 256)
    for _ in range(epochs):
        for _step in range(steps):
            if mode == "naive":
                ix = rng.choice(len(Xt), size=min(256, len(Xt)), replace=False)
                loss = nn.functional.binary_cross_entropy_with_logits(net.instance_logits(Xt[ix]), st[ix])
            else:  # nnpu
                pb = rng.choice(pos_ix, size=min(128, len(pos_ix)), replace=len(pos_ix) < 128)
                ub = rng.choice(unl_ix, size=min(256, len(unl_ix)), replace=False)
                loss = nnpu_loss(net.instance_logits(Xt[pb]), net.instance_logits(Xt[ub]), prior)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def score_neural(preprocess, net: FlowNet, X: pd.DataFrame) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net.instance_logits(_tensor(preprocess, X))).numpy()
