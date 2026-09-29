"""
The trained policy for the live engine: a numpy forward pass, no torch needed.

Saved as one .npz holding the actor and critic weights, the observation
normalisation, and a JSON metadata blob (training window, environment rules,
evaluation report, promotion gate). The live engine reloads it when the file
changes, so a retrain on the host is picked up without a restart.
"""
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "rl")
POLICY_PATH = os.path.join(MODEL_DIR, "policy.npz")


def export(model, norm: Dict[str, np.ndarray], meta: Dict[str, Any], path: str = POLICY_PATH):
    """Writes a torch ActorCritic as numpy weights (atomically)."""
    arrays = {}
    for net in ("actor", "critic"):
        lin = [m for m in getattr(model, net) if hasattr(m, "weight")]
        for i, m in enumerate(lin):
            arrays[f"{net}_W{i}"] = m.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"{net}_b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    for k, v in norm.items():
        arrays[f"norm_{k}"] = np.asarray(v, dtype=np.float32)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta, default=_json_default).encode(), dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_into_torch(model, path: str = POLICY_PATH):
    """Warm start: copies saved weights into a torch ActorCritic of the same shape."""
    import torch
    z = np.load(path)
    for net in ("actor", "critic"):
        lin = [m for m in getattr(model, net) if hasattr(m, "weight")]
        for i, m in enumerate(lin):
            m.weight.data.copy_(torch.as_tensor(z[f"{net}_W{i}"]))
            m.bias.data.copy_(torch.as_tensor(z[f"{net}_b{i}"]))


class NumpyPolicy:
    def __init__(self, path: str = POLICY_PATH):
        self.path = path
        self.actor: List[Tuple[np.ndarray, np.ndarray]] = []
        self.critic: List[Tuple[np.ndarray, np.ndarray]] = []
        self.norm: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, Any] = {}
        self._mtime = 0.0
        self._lock = threading.Lock()
        self.error: Optional[str] = None

    def refresh(self) -> bool:
        """(Re)loads the file when it changed. True when a policy is loaded."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self.actor, self.meta = [], {}
            return False
        if self.actor and mtime == self._mtime:
            return True
        with self._lock:
            try:
                z = np.load(self.path)
                layers = lambda net: [(z[f"{net}_W{i}"], z[f"{net}_b{i}"])
                                      for i in range(sum(1 for k in z.files if k.startswith(f"{net}_W")))]
                self.actor, self.critic = layers("actor"), layers("critic")
                self.norm = {k[5:]: z[k] for k in z.files if k.startswith("norm_")}
                self.meta = json.loads(bytes(z["meta_json"]).decode())
                self._mtime = mtime
                self.error = None
            except Exception as e:
                self.actor, self.meta = [], {}
                self.error = f"{type(e).__name__}: {e}"[:200]
                return False
        return True

    @property
    def loaded(self) -> bool:
        return bool(self.actor)

    @staticmethod
    def _forward(layers, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(layers):
            x = x @ W.T + b
            if i < len(layers) - 1:
                x = np.tanh(x)
        return x

    def probs(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """P(flat), P(long) per row, with forbidden actions at 0."""
        logits = self._forward(self.actor, np.atleast_2d(obs).astype(np.float32))
        if mask is not None:
            logits = np.where(np.atleast_2d(mask), logits, -1e9)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def value(self, obs: np.ndarray) -> np.ndarray:
        return self._forward(self.critic, np.atleast_2d(obs).astype(np.float32))[:, 0]

    @property
    def approved(self) -> bool:
        return bool((self.meta.get("gate") or {}).get("approved"))
