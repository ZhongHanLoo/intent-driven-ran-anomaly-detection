"""Detector architectures.

Supervised (emit logits for BCE): LstmDetector (Paper 1 Table VII best config),
TcnDetector and TransformerDetector (built within Paper 1's Table VI search
spaces; exact winning configs unpublished — documented deviation).
Unsupervised: AeDetector — dense autoencoder trained on benign-only windows;
anomaly score = per-window reconstruction MSE (needs no attack examples).
"""

from __future__ import annotations

import torch
from torch import nn


class LstmDetector(nn.Module):
    """LSTM(14->32) + linear head. 6,177 params (== Keras 6,049 + double-bias 128)."""

    def __init__(self, n_features: int = 14, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # (B, w, f) -> (B,) logits
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):  # (B, C, T)
        h = self.drop(self.act(self.conv1(x)[..., : -self.pad or None]))
        h = self.drop(self.act(self.conv2(h)[..., : -self.pad or None]))
        return self.act(h + self.down(x))


class TcnDetector(nn.Module):
    """Temporal Convolutional Network: 2 causal residual blocks (32 filters,
    kernel 3, dilations 1 and 2, dropout 0.1) + linear head on the last step."""

    def __init__(self, n_features: int = 14, channels: int = 32, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.b1 = _CausalBlock(n_features, channels, kernel, 1, dropout)
        self.b2 = _CausalBlock(channels, channels, kernel, 2, dropout)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):  # (B, w, f)
        h = self.b2(self.b1(x.transpose(1, 2)))  # (B, C, T)
        return self.head(h[:, :, -1]).squeeze(-1)


class TransformerDetector(nn.Module):
    """Input projection 14->32, learned positional embedding, one encoder layer
    (2 heads, feed-forward 64, dropout 0.1), head on the last position."""

    def __init__(self, n_features: int = 14, d_model: int = 32, window: int = 3):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=64, dropout=0.1, batch_first=True
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # (B, w, f)
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h[:, -1, :]).squeeze(-1)


class AeDetector(nn.Module):
    """Dense autoencoder over flattened windows (w*f -> 32 -> 8 -> 32 -> w*f).
    Trained on BENIGN windows only; score(x) = per-window reconstruction MSE."""

    def __init__(self, n_features: int = 14, window: int = 3, bottleneck: int = 8):
        super().__init__()
        d = n_features * window
        self.window, self.n_features = window, n_features
        self.encdec = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, 32), nn.ReLU(), nn.Linear(32, d),
        )

    def forward(self, x):  # (B, w, f) -> reconstruction (B, w, f)
        b = x.shape[0]
        return self.encdec(x.reshape(b, -1)).reshape(b, self.window, self.n_features)

    def score(self, x):  # (B, w, f) -> (B,) reconstruction MSE
        return ((self.forward(x) - x) ** 2).mean(dim=(1, 2))


def make_model(name: str, window: int = 3, n_features: int = 14) -> nn.Module:
    if name == "lstm":
        return LstmDetector(n_features)
    if name == "tcn":
        return TcnDetector(n_features)
    if name == "transformer":
        return TransformerDetector(n_features, window=window)
    if name == "ae":
        return AeDetector(n_features, window=window)
    raise ValueError(f"unknown model: {name}")
