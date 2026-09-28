"""
The trade scorer: a small LSTM that reads the last hour of one-minute bars at
the moment a strategy wants to enter and estimates whether that trade will make
money AFTER spread and fees.

It is the simplest form of reinforcement learning, a contextual bandit: one
decision per trade (take it or skip it), rewarded by the trade's net result.
It learns from every closed trade, live ones weighted above backtested ones and
losses above wins, so it learns most from mistakes (ml/train.py).

About 35k parameters. Trained on the GPU (python -m ml.train); scored on the CPU
inside the bot, where one batch of candidates takes well under a millisecond
and never touches the tick path: the portfolio manager scores its entry
candidates once per cycle (engine/portfolio_manager.py).
"""
import logging
import os
import threading
from typing import Dict, List, Optional

import numpy as np

from ml.features import N_BAR_FEATURES, STRATEGY_VOCAB

logger = logging.getLogger("tradeflow.ml")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trade_scorer.pt")


def build(hidden: int = 48, layers: int = 1, strat_dim: int = 8):
    import torch
    from torch import nn

    class TradeScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_BAR_FEATURES, hidden, num_layers=layers, batch_first=True,
                                dropout=0.1 if layers > 1 else 0.0)
            self.strat = nn.Embedding(len(STRATEGY_VOCAB), strat_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden + strat_dim + 1, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 2))          # [logit of a net win, expected net return in %]

        def forward(self, bars, strat, crypto):
            _, (h, _) = self.lstm(bars)
            z = torch.cat([h[-1], self.strat(strat), crypto.unsqueeze(1)], dim=1)
            return self.head(z)

    return TradeScorer()


class Scorer:
    """
    Loads the trained model for CPU inference and reloads it when the file
    changes, so a retrain on the host is picked up without restarting the bot.
    Never raises: without torch or a model file, score() returns None and the
    bot trades exactly as it would without it.
    """

    def __init__(self, path: str = MODEL_PATH):
        self.path = path
        self.model = None
        self.meta: Dict = {}
        self._mtime = 0.0
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    def _load_if_changed(self) -> bool:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self.model = None
            return False
        if self.model is not None and mtime == self._mtime:
            return True
        with self._lock:
            try:
                import torch
                torch.set_num_threads(1)     # the bot's event loop shares this CPU
                ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
                m = build(**ckpt["arch"])
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                self.model, self.meta, self._mtime = m, ckpt["meta"], mtime
                self.last_error = None
                logger.info("Loaded trade scorer trained %s (holdout: %s)",
                            self.meta.get("trained_at"), self.meta.get("holdout"))
            except Exception as e:
                self.model = None
                self.last_error = f"{type(e).__name__}: {e}"[:200]
                logger.warning("Trade scorer not loaded: %s", self.last_error)
        return self.model is not None

    @property
    def ready(self) -> bool:
        return self._load_if_changed()

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.5))

    def score(self, items: List[Dict]) -> Optional[List[Dict[str, float]]]:
        """
        items: [{"bars": (WINDOW, F) array, "strategy": name, "crypto": bool}].
        Returns [{"p_win", "exp_ret_pct"}] in the same order, or None.
        """
        if not items or not self._load_if_changed():
            return None
        try:
            import torch
            mu = np.asarray(self.meta["feat_mean"], dtype=np.float32)
            sd = np.asarray(self.meta["feat_std"], dtype=np.float32)
            from ml.features import strategy_index
            x = torch.from_numpy(np.stack([(it["bars"] - mu) / sd for it in items]).astype(np.float32))
            s = torch.tensor([strategy_index(it["strategy"]) for it in items], dtype=torch.long)
            c = torch.tensor([1.0 if it["crypto"] else 0.0 for it in items])
            with torch.inference_mode():
                out = self.model(x, s, c)
            p = torch.sigmoid(out[:, 0]).tolist()
            r = (out[:, 1] * float(self.meta.get("ret_scale", 1.0))).tolist()
            return [{"p_win": round(pi, 4), "exp_ret_pct": round(ri, 4)} for pi, ri in zip(p, r)]
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            logger.warning("Trade scorer failed: %s", self.last_error)
            return None

    def status(self) -> Dict:
        return {"loaded": self.model is not None, "path": self.path, "error": self.last_error,
                "trained_at": self.meta.get("trained_at"), "threshold": self.meta.get("threshold"),
                "holdout": self.meta.get("holdout"), "samples": self.meta.get("samples")}


scorer = Scorer()
