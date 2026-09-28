"""
Train the trade scorer on the GPU, from backtested trades plus the bot's own
live experience, and save it only if it beats "take every trade" on trades it
has never seen.

    cd api && python3 -m ml.train                   # 30 days of bars, the watchlist
    python3 -m ml.train --days 60 --epochs 40
    python3 -m ml.train --live-only                 # just the bot's own trades
    python3 -m ml.train --force                     # save even if it does not help

Learning from mistakes: every sample is weighted, live trades LIVE_WEIGHT x
backtested ones (they are what the bot actually does), and losing trades
LOSS_WEIGHT x winners (the mistakes to avoid).

Honest evaluation: samples are split by TIME -- the oldest 70% train, the next
15% choose the epoch and the skip threshold, the newest 15% are the test. The
model is saved only when skipping the trades it scores under the threshold
would have raised the test set's net P&L. Otherwise it reports that it found
nothing, and the bot keeps trading without it.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

LIVE_WEIGHT = 3.0
LOSS_WEIGHT = 1.5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _bt_job(args):
    symbol, bars, strat_names, notional = args
    from backtest import sim
    from engine.strategies import registry
    from ml.features import strategy_index
    tape = sim.Tape(symbol, bars, sim.Costs())
    klass = registry.asset_class(symbol)
    out = []
    with sim.overrides(sim.VARIANTS["current"]):
        for name in strat_names:
            strat = registry.get(name)
            if strat is None or not registry.is_compatible(name, klass):
                continue
            for t in sim.run(tape, strat, notional, collect_features=True):
                f = getattr(t, "features", None)
                if f is None:
                    continue
                out.append({"bars": f, "strat": strategy_index(name), "crypto": tape.crypto,
                            "ret": t.pnl / notional * 100.0, "t": t.entry_minute * 60.0, "live": False})
    return out


def backtest_samples(symbols: List[str], days: int, notional: float, workers: int) -> List[Dict]:
    from backtest import data
    from engine.strategies import registry
    bars = data.load_many(symbols, days)
    names = [s.name for s in registry.available()]
    out: List[Dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(_bt_job, [(s, b, names, notional) for s, b in bars.items()]):
            out.extend(rows)
    return out


def live_samples() -> List[Dict]:
    from ml.experience import EXPERIENCE_PATH
    from ml.features import strategy_index, WINDOW, N_BAR_FEATURES
    out = []
    try:
        with open(EXPERIENCE_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    b = np.asarray(r["bars"], dtype=np.float32)
                    if b.shape != (WINDOW, N_BAR_FEATURES):
                        continue
                    out.append({"bars": b, "strat": strategy_index(r.get("strategy")),
                                "crypto": bool(r.get("crypto")), "ret": float(r["ret_pct"]),
                                "t": float(r["t"]), "live": True})
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _tensors(rows, mu, sd, device):
    import torch
    x = torch.tensor(np.stack([(r["bars"] - mu) / sd for r in rows]), dtype=torch.float32, device=device)
    s = torch.tensor([r["strat"] for r in rows], dtype=torch.long, device=device)
    c = torch.tensor([1.0 if r["crypto"] else 0.0 for r in rows], device=device)
    ret = torch.tensor([r["ret"] for r in rows], dtype=torch.float32, device=device)
    w = torch.tensor([(LIVE_WEIGHT if r["live"] else 1.0) * (LOSS_WEIGHT if r["ret"] <= 0 else 1.0)
                      for r in rows], dtype=torch.float32, device=device)
    return x, s, c, ret, w


def _predict(model, x, s, c, batch=4096):
    import torch
    model.eval()
    with torch.inference_mode():
        return torch.cat([torch.sigmoid(model(x[i:i + batch], s[i:i + batch], c[i:i + batch])[:, 0])
                          for i in range(0, len(x), batch)]).cpu().numpy()


def _best_threshold(p: np.ndarray, ret: np.ndarray) -> float:
    """The skip threshold that maximises the net return of the trades kept."""
    best_t, best = 0.0, ret.sum()
    for t in np.quantile(p, np.linspace(0.05, 0.9, 35)):
        kept = ret[p >= t].sum()
        if kept > best:
            best_t, best = float(t), kept
    return best_t


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(rows: List[Dict], epochs: int = 30, hidden: int = 48, lr: float = 2e-3,
          batch: int = 512, seed: int = 0, device: str = None, verbose: bool = True) -> Dict:
    """Trains on time-ordered rows; returns {"model", "arch", "meta", "report"}."""
    import torch
    from torch import nn
    from ml.model import build
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows = sorted(rows, key=lambda r: r["t"])
    n = len(rows)
    a, b = int(n * 0.70), int(n * 0.85)
    tr, va, te = rows[:a], rows[a:b], rows[b:]
    allb = np.concatenate([r["bars"] for r in tr])
    mu, sd = allb.mean(axis=0), allb.std(axis=0) + 1e-6
    ret_scale = float(np.std([r["ret"] for r in tr]) or 1.0)

    Xtr, Str, Ctr, Rtr, Wtr = _tensors(tr, mu, sd, device)
    Xva, Sva, Cva, Rva, _ = _tensors(va, mu, sd, device)
    Xte, Ste, Cte, Rte, _ = _tensors(te, mu, sd, device)
    Ytr = (Rtr > 0).float()

    arch = {"hidden": hidden, "layers": 1, "strat_dim": 8}
    model = build(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none")
    best_state, best_val, stale = None, -np.inf, 0
    rva = Rva.cpu().numpy()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            out = model(Xtr[j], Str[j], Ctr[j])
            loss = ((bce(out[:, 0], Ytr[j]) + 0.5 * huber(out[:, 1], Rtr[j] / ret_scale)) * Wtr[j]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(j)
        pva = _predict(model, Xva, Sva, Cva)
        # Validation objective: net return kept at the best threshold.
        val = rva[pva >= _best_threshold(pva, rva)].sum()
        if verbose:
            print(f"  epoch {ep + 1:2d}  loss {tot / len(Xtr):.4f}  val kept net {val:+.1f}%  "
                  f"auc {_auc(pva, (rva > 0).astype(int)):.3f}")
        if val > best_val:
            best_val, stale = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best_state)
    pva = _predict(model, Xva, Sva, Cva)
    thr = _best_threshold(pva, rva)
    pte = _predict(model, Xte, Ste, Cte)
    rte = Rte.cpu().numpy()
    kept = pte >= thr
    report = {
        "samples": {"train": len(tr), "val": len(va), "test": len(te),
                    "live": sum(r["live"] for r in rows)},
        "device": device, "seconds": round(time.time() - t0, 1),
        "threshold": round(thr, 4),
        "test_auc": round(_auc(pte, (rte > 0).astype(int)), 4),
        "test_trades_all": int(len(rte)), "test_trades_kept": int(kept.sum()),
        # Net return summed over trades, % of notional per trade: x $10 = $ per $1,000 trade.
        "test_net_pct_all": round(float(rte.sum()), 2),
        "test_net_pct_kept": round(float(rte[kept].sum()), 2),
        "test_win_rate_all": round(float((rte > 0).mean() * 100), 1),
        "test_win_rate_kept": round(float((rte[kept] > 0).mean() * 100), 1) if kept.any() else 0.0,
    }
    report["improves"] = report["test_net_pct_kept"] > report["test_net_pct_all"]
    meta = {"feat_mean": mu.tolist(), "feat_std": sd.tolist(), "threshold": thr, "ret_scale": ret_scale,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": report["samples"],
            "holdout": {k: report[k] for k in ("test_auc", "test_net_pct_all", "test_net_pct_kept",
                                               "test_trades_all", "test_trades_kept")}}
    return {"model": model.cpu(), "arch": arch, "meta": meta, "report": report}


def save(result: Dict, path: str = None):
    import torch
    from ml.model import MODEL_PATH, MODEL_DIR
    path = path or MODEL_PATH
    os.makedirs(os.path.dirname(path) or MODEL_DIR, exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"arch": result["arch"], "state_dict": result["model"].state_dict(),
                "meta": result["meta"]}, tmp)
    os.replace(tmp, path)      # atomic: the bot never loads a half-written file


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ml.train")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="save even if the test set does not improve")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)

    from core.state import state
    rows = live_samples()
    print(f"Live experience: {len(rows)} closed trades")
    if not a.live_only:
        symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or sorted(state.watchlist)
        print(f"Replaying {a.days} days of bars for {len(symbols)} symbols to collect backtested trades...")
        bt = backtest_samples(symbols, a.days, a.notional, a.workers)
        print(f"Backtested trades: {len(bt)}")
        rows += bt
    if len(rows) < 200:
        raise SystemExit(f"Only {len(rows)} samples: not enough to train on.")

    res = train(rows, epochs=a.epochs, hidden=a.hidden)
    r = res["report"]
    print("\nHoldout (newest 15% of trades, never trained on):")
    print(f"  trades {r['test_trades_all']} -> kept {r['test_trades_kept']} at p(win) >= {r['threshold']:.3f}")
    print(f"  net    {r['test_net_pct_all']:+.1f}% -> {r['test_net_pct_kept']:+.1f}%  "
          f"(sum of per-trade returns; x$10 = dollars on $1,000 trades)")
    print(f"  win    {r['test_win_rate_all']:.1f}% -> {r['test_win_rate_kept']:.1f}%   AUC {r['test_auc']:.3f}")
    print(f"  trained on {r['device']} in {r['seconds']}s, {r['samples']}")
    if r["improves"] or a.force:
        save(res)
        print(f"\nSaved to models/trade_scorer.pt. The bot reloads it within one manager cycle.")
    else:
        print("\nNot saved: skipping trades by this model would not have helped on unseen data.")
    return r


if __name__ == "__main__":
    main()
