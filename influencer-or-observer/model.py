"""
TabularMLP — a deep MLP for the structured (tabular) feature set.

Architecture (single-logit + BCEWithLogitsLoss), per the approved plan:
    Linear(in -> 256) -> BatchNorm1d -> GELU -> Dropout(0.30)
    Linear(256 -> 128) -> BatchNorm1d -> GELU -> Dropout(0.30)
    Linear(128 -> 64)  -> BatchNorm1d -> GELU -> Dropout(0.20)
    Linear(64 -> 1)                                  # raw logit

Design notes:
- BatchNorm1d after Linear, before activation: stabilizes training on the
  standardized inputs and permits a higher learning rate.
- GELU over ReLU for smoother gradients (ReLU is a fine swap).
- Single logit + BCEWithLogitsLoss is numerically stable; sigmoid(logit) is the
  probability used for thresholding and ensembling.
- No categorical embeddings: `lang` is near-constant in this data and `source`
  is captured by the fold-safe scalar target encoding rather than a sparse
  embedding over ~190 classes.
- HONEST CAVEAT: gradient-boosted trees (the repo's LightGBM subs) typically
  match or beat an MLP on this kind of tabular metadata. The user asked for a
  deep-learning model, so this MLP is built as strong as is reasonable; expect
  OOF accuracy ~89-91%, comparable to sub19's ~90.5%.
"""
import torch
import torch.nn as nn

import config


class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=None, dropout=None):
        super().__init__()
        hidden = hidden if hidden is not None else config.HIDDEN
        dropout = dropout if dropout is not None else config.DROPOUT

        layers = []
        prev = in_dim
        for h, p in zip(hidden, dropout):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p))
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Return a flat [N] logit vector (matches BCEWithLogitsLoss targets).
        return self.net(x).squeeze(-1)


def build_model(in_dim, hidden=None, dropout=None):
    """Factory so train.py constructs the model in one call."""
    return TabularMLP(in_dim, hidden=hidden, dropout=dropout)
