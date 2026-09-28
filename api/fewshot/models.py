"""
A tiny forecaster: the input the RL policy sees -> next-H-minute return in bps.

Small on purpose (about 20k weights): minute returns are almost all noise, and
a model that can adapt in a few gradient steps must not be able to memorise.
Trained with a Huber loss on the target scaled to unit spread, so the fat tails
of minute returns do not dominate.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from fewshot.data import D_IN


class Forecaster(nn.Module):
    def __init__(self, d_in: int = D_IN, hidden: Tuple[int, ...] = (64, 32)):
        super().__init__()
        layers, last = [], d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x):                # -> prediction in units of y_scale
        return self.net(x).squeeze(-1)


def loss_fn(pred, y_scaled):
    return nn.functional.huber_loss(pred, y_scaled, delta=1.0)


def to_numpy(model: Forecaster) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [(m.weight.detach().cpu().numpy().astype(np.float32), m.bias.detach().cpu().numpy().astype(np.float32))
            for m in model.net if isinstance(m, nn.Linear)]


def numpy_forward(layers: List[Tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = np.tanh(x)
    return x[..., 0]
