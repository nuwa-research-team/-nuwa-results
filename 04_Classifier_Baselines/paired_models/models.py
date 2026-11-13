# -*- coding: utf-8 -*-
"""Six downstream traffic classifiers used in the Nüwa paper.

All models consume a (B, 1, 512) float sequence of signed packet lengths
normalised by 1600, and output `num_classes` logits.
"""
import math

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes, in_len=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1))


class RNN(nn.Module):
    def __init__(self, num_classes, cell="lstm", hidden=128, layers=2):
        super().__init__()
        cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = cls(1, hidden, num_layers=layers, batch_first=True,
                       bidirectional=False, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, T, 1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TST(nn.Module):
    """Time-Series Transformer classifier."""

    def __init__(self, num_classes, d_model=128, layers=2, heads=4):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.pos(self.proj(x.transpose(1, 2)))
        h = self.encoder(h)
        return self.fc(h.mean(1))


class SDAE(nn.Module):
    """Stacked denoising auto-encoder with a classification head on the latent."""

    def __init__(self, num_classes, in_len=512, latent=128):
        super().__init__()
        self.in_len = in_len
        self.enc = nn.Sequential(nn.Linear(in_len, 256), nn.ReLU(),
                                 nn.Linear(256, latent), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(),
                                 nn.Linear(256, in_len))
        self.fc = nn.Linear(latent, num_classes)

    def forward(self, x, recon=False):
        h = x.flatten(1)
        if h.size(1) < self.in_len:            # dynamic padding support
            h = nn.functional.pad(h, (0, self.in_len - h.size(1)))
        elif h.size(1) > self.in_len:
            h = h[:, : self.in_len]
        z = self.enc(h)
        if recon:
            return self.fc(z), self.dec(z)
        return self.fc(z)


class FSNet(nn.Module):
    """FS-Net style bi-directional GRU with attention pooling (simplified)."""

    def __init__(self, num_classes, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.att = nn.Linear(2 * hidden, 1)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        h, _ = self.gru(x.transpose(1, 2))
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)
        return self.fc((h * w.unsqueeze(-1)).sum(1))


def build(name, num_classes):
    name = name.lower()
    if name == "cnn":
        return CNN(num_classes)
    if name == "lstm":
        return RNN(num_classes, "lstm")
    if name == "gru":
        return RNN(num_classes, "gru")
    if name == "tst":
        return TST(num_classes)
    if name == "sdae":
        return SDAE(num_classes)
    if name in ("fsnet", "fs-net"):
        return FSNet(num_classes)
    raise ValueError(name)
