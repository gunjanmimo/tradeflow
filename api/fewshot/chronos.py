"""
Chronos-Bolt (Ansari et al., 2024; Amazon, Apache-2.0) as a lightweight
pretrained base. The tiny model has ~9M weights and was pretrained on a large
corpus of real and synthetic time series; it forecasts without any training on
our data (zero-shot).

For every decision bar it reads the symbol's last CONTEXT one-minute closes
(crossing into earlier sessions) and forecasts the next `horizon` minutes. The
signal is the forecast median's log return over the horizon, in bps.

Few-shot use: each day, a linear calibration y ~ a + b * forecast is fitted on
the last K days (ridge-shrunk towards "no signal"), so the pretrained forecast
is re-scaled to what it has recently been worth.

Optional dependency: pip install chronos-forecasting (weights download once
from Hugging Face, ~35 MB).
"""
import hashlib
import json
import os
from typing import Optional

import numpy as np

from rl import features as F

MODEL_ID = "amazon/chronos-bolt-tiny"
CONTEXT = 512
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "fewshot")


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("chronos") is not None


def _pipeline(device: str):
    import torch
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.float32)


def forecasts(ss, sessions: np.ndarray, horizon: int, device: str = "cuda", batch: int = 4096,
              cache_key: Optional[str] = None, verbose: bool = True) -> np.ndarray:
    """
    Forecast log return (bps) over `horizon` minutes at every decision bar of the
    given sessions. Returns [len(sessions), N_STEPS].
    """
    import torch
    key = hashlib.sha1(json.dumps({"k": cache_key, "h": horizon, "m": MODEL_ID, "c": CONTEXT,
                                   "s": [int(sessions[0]), int(sessions[-1]), len(sessions)]}).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"chronos_{key}.npy")
    if cache_key and os.path.exists(path):
        return np.load(path)
    pipe = _pipeline(device)
    # each symbol's sessions end to end: one continuous close series per symbol
    pos = np.empty(len(ss), dtype=np.int64)
    series = {}
    for m in np.unique(ss.sym):
        idx = np.flatnonzero(ss.sym == m)
        series[m] = ss.c[idx].reshape(-1).astype(np.float32)
        pos[idx] = np.arange(len(idx))
    dec = np.asarray(F.DECISIONS)
    pairs = [(s, k) for s in sessions for k in range(len(dec))]
    out = np.empty(len(pairs), dtype=np.float32)
    ar = np.arange(-CONTEXT + 1, 1)
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        ctx = np.empty((len(chunk), CONTEXT), dtype=np.float32)
        last = np.empty(len(chunk), dtype=np.float32)
        for j, (s, k) in enumerate(chunk):
            ser = series[ss.sym[s]]
            end = pos[s] * F.SESSION_BARS + dec[k]
            ix = np.clip(end + ar, 0, None)          # repeats the first close when history is short
            ctx[j] = ser[ix]
            last[j] = ser[end]
        with torch.no_grad():
            q, _ = pipe.predict_quantiles(torch.from_numpy(ctx), prediction_length=horizon,
                                          quantile_levels=[0.5])
        med = q[:, horizon - 1, 0].cpu().numpy()
        out[i:i + len(chunk)] = np.log(np.maximum(med, 1e-9) / last) * 1e4
        if verbose and (i // batch) % 20 == 0:
            print(f"    chronos {i + len(chunk):,}/{len(pairs):,} forecasts")
    res = out.reshape(len(sessions), len(dec))
    if cache_key:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, res)
    return res


def calibrate(pred: np.ndarray, y: np.ndarray) -> tuple:
    """
    (a, b) of y ~ a + b * pred, with the slope shrunk by max(0, 1 - 1/t^2):
    a relation no stronger than noise (t ~ 1) fits b ~ 0, a strong one keeps
    almost all of it. The intercept is the support days' mean return.
    """
    pred, y = np.asarray(pred, np.float64), np.asarray(y, np.float64)
    n = len(pred)
    p = pred - pred.mean()
    v = float((p * p).sum())
    if n < 10 or v <= 0:
        return float(y.mean()) if n else 0.0, 0.0
    b = float((p * (y - y.mean())).sum()) / v
    resid = (y - y.mean()) - b * p
    se = np.sqrt(float((resid * resid).sum()) / (n - 2) / v)
    t = b / se if se > 0 else 0.0
    b *= max(0.0, 1.0 - 1.0 / (t * t)) if t != 0 else 0.0
    return float(y.mean() - b * pred.mean()), b
