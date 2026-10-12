"""
ChromatogramCNN (ported from chroma-dcnn) as the deep counterpart.

Architecture:
  1. Per-bin spectral projection  Linear(mz_max → channels)  →  [B, N_bins, channels]
  2. Three dilated 1D ResBlocks over RT (kernel=7, dilation=1/2/4)
  3. Dual temporal pooling: global max-pool ∥ soft attention-pool → [B, 2·channels]
     — this pooled vector is the embedding used by kNN / Mahalanobis scores
  4. Linear classification head

Trained from scratch with class-weighted cross-entropy, AdamW + cosine schedule
(chroma-dcnn's from_scratch settings), and random RT-shift augmentation to
mimic retention-time drift between runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class ChromaCNNConfig:
    mz_max: int = 1000
    cnn_channels: int = 128
    kernel_size: int = 7
    num_classes: int = 4
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_shift: int = 3          # RT bins of random shift augmentation


class _ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad        = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        self.act   = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x   = self.act(self.bn1(self.conv1(x)))
        x   = self.drop(x)
        x   = self.bn2(self.conv2(x))
        return self.act(res + x)


class _AttentionPool1d(nn.Module):
    """Soft attention pooling over the temporal (RT) dimension.  x: [B,C,N] → [B,C]"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x_t = x.transpose(1, 2)                    # [B, N, C]
        w = F.softmax(self.score(x_t), dim=1)      # [B, N, 1]
        return (x_t * w).sum(dim=1)                # [B, C]


class ChromatogramCNN(nn.Module):
    def __init__(self, config: ChromaCNNConfig) -> None:
        super().__init__()
        ch = config.cnn_channels
        self.spec_proj = nn.Linear(config.mz_max, ch)
        k, d = config.kernel_size, config.dropout
        self.cnn = nn.Sequential(
            _ResBlock1d(ch, k, dilation=1, dropout=d),
            _ResBlock1d(ch, k, dilation=2, dropout=d),
            _ResBlock1d(ch, k, dilation=4, dropout=d),
        )
        self.attn_pool = _AttentionPool1d(ch)
        self.head = nn.Linear(ch * 2, config.num_classes)

    def embed(self, x: Tensor) -> Tensor:
        """[B, N_bins, mz_max] → [B, 2·channels]"""
        B, N, mz = x.shape
        emb = self.spec_proj(x.reshape(B * N, mz)).view(B, N, -1)  # [B, N, ch]
        x   = self.cnn(emb.transpose(1, 2))                         # [B, ch, N]
        return torch.cat([x.max(dim=-1).values, self.attn_pool(x)], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.embed(x))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rt_shift(x: Tensor, max_shift: int) -> Tensor:
    """Shift each chromatogram by a random number of RT bins, zero-filling the gap."""
    if max_shift == 0:
        return x
    out = torch.zeros_like(x)
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],))
    n = x.shape[1]
    for i, s in enumerate(shifts.tolist()):
        if s >= 0:
            out[i, s:] = x[i, : n - s]
        else:
            out[i, :s] = x[i, -s:]
    return out


class DeepCNN:
    """sklearn-style wrapper so the experiment loop treats it like a baseline."""

    def __init__(self, seed: int = 0, train_cfg: TrainConfig | None = None,
                 model_cfg: ChromaCNNConfig | None = None) -> None:
        self.seed = seed
        self.train_cfg = train_cfg or TrainConfig()
        self.model_cfg = model_cfg or ChromaCNNConfig()
        self.device = _device()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepCNN":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        cfg = self.train_cfg
        n_classes = self.model_cfg.num_classes
        self.model = ChromatogramCNN(self.model_cfg).to(self.device)

        counts = np.bincount(y, minlength=n_classes)
        weight = torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)),
                              dtype=torch.float32, device=self.device)
        opt = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).long()
        self.model.train()
        for _ in range(cfg.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                if len(idx) < 2:   # BatchNorm needs > 1 sample
                    continue
                xb = _rt_shift(X_t[idx], cfg.max_shift).to(self.device)
                loss = F.cross_entropy(self.model(xb), y_t[idx].to(self.device), weight=weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def outputs(self, X: np.ndarray, batch_size: int = 32) -> dict:
        self.model.eval()
        embs = []
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).float().to(self.device)
            embs.append(self.model.embed(xb))
        emb = torch.cat(embs)
        logits = self.model.head(emb)
        return {
            "probs": F.softmax(logits, dim=1).cpu().numpy(),
            "logits": logits.cpu().numpy(),
            "emb": emb.cpu().numpy(),
        }
