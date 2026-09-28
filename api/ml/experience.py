"""
Live experience: what the bot saw when it entered each trade, and what that
trade finally made. This is what the trade scorer learns from.

  on_entry   the portfolio manager's entry: the bar window it scored, the
             strategy that fired, and the model's score (if one is loaded)
  on_close   the position's whole result -- final sale, profit harvests and
             trims -- appended as one line to
             data/ml/experience.jsonl

Nothing here is on the tick path: an entry is recorded once per trade, a close
once per trade, and the file append is a few hundred bytes. Pending entries are
also kept on disk so a restart does not orphan open trades.
"""
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from core.config import settings

logger = logging.getLogger("tradeflow.ml")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ml")
EXPERIENCE_PATH = os.path.join(DATA_DIR, "experience.jsonl")
PENDING_PATH = os.path.join(DATA_DIR, "pending.json")


def net_result(pos: Dict[str, Any], final_pnl: float, crypto: bool) -> Dict[str, float]:
    """The position's whole result in dollars and % of what was invested."""
    invested = float(pos.get("invested_dollars") or 0.0)
    if invested <= 0:
        invested = float(pos.get("qty") or 0.0) * float(pos.get("avg_entry_price") or 0.0)
    gross = (float(final_pnl) + float(pos.get("harvested_income") or 0.0)
             + float(pos.get("trimmed_pnl") or 0.0))
    fees = 0.0
    net = gross - fees
    return {"net_pnl": round(net, 4), "ret_pct": round(net / invested * 100, 4) if invested > 0 else 0.0,
            "invested": round(invested, 2)}


class Experience:
    def __init__(self):
        self.pending: Dict[str, Dict[str, Any]] = {}
        self.recorded = 0
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(PENDING_PATH) as f:
                self.pending = json.load(f)
        except (OSError, ValueError):
            self.pending = {}

    def _save_pending(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            tmp = PENDING_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.pending, f)
            os.replace(tmp, PENDING_PATH)
        except OSError as e:
            logger.debug("Could not save pending ML entries: %s", e)

    def on_entry(self, symbol: str, bars: Optional[np.ndarray], strategy: str, crypto: bool,
                 score: Optional[Dict[str, float]] = None):
        if bars is None or not settings.ML_RECORD_EXPERIENCE:
            return
        with self._lock:
            self._load()
            self.pending[symbol] = {"t": time.time(), "symbol": symbol, "strategy": strategy,
                                    "crypto": bool(crypto), "bars": np.round(bars, 6).tolist(),
                                    "score": score}
            self._save_pending()

    def on_close(self, symbol: str, pos: Dict[str, Any], final_pnl: float):
        with self._lock:
            self._load()
            entry = self.pending.pop(symbol, None)
            if entry is None:
                return
            self._save_pending()
        entry.update(net_result(pos, final_pnl, entry["crypto"]))
        entry.update(closed_at=time.time(), source="live",
                     held_s=round(time.time() - entry["t"], 1))
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(EXPERIENCE_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self.recorded += 1
        except OSError as e:
            logger.warning("Could not record ML experience for %s: %s", symbol, e)

    def status(self) -> Dict[str, Any]:
        try:
            with open(EXPERIENCE_PATH) as f:
                total = sum(1 for _ in f)
        except OSError:
            total = 0
        return {"open_entries": len(self.pending), "recorded_this_run": self.recorded,
                "recorded_total": total}


experience = Experience()
