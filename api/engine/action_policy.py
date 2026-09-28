"""
4-action policy distribution for an open position: [BUY, HOLD, SELL, CLOSE].

CLOSE is not one more weight competing in a pool. It is the exit probability
itself (bracket proximity, reversal risk, strategy read) and takes its share of
the 100% first; BUY / HOLD / SELL split what is left in proportion to their
weights. Pooling all four weights and normalizing, as before, capped CLOSE near
40% even at the stop and squeezed every position into the same distribution.

Invariant the UI relies on: CLOSE reads 100% exactly when the exit is fired.
Short of that it is held at CLOSE_CAP, and the action label is CLOSE only then.
"""
from typing import Dict, Tuple

ACTION_SPACE = ["BUY", "HOLD", "SELL", "CLOSE"]
CLOSE_CAP = 0.99


def distribute(buy_w: float, hold_w: float, sell_w: float, close_p: float,
               closing: bool) -> Tuple[Dict[str, float], str]:
    """Returns ({action: prob} summing to 1.0, resolved action)."""
    if closing:
        return {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0, "CLOSE": 1.0}, "CLOSE"

    close_p = min(max(float(close_p), 0.0), CLOSE_CAP)
    weights = {"BUY": max(float(buy_w), 0.0), "HOLD": max(float(hold_w), 0.0),
               "SELL": max(float(sell_w), 0.0)}
    total = sum(weights.values())
    if total <= 0:
        weights, total = {"BUY": 0.0, "HOLD": 1.0, "SELL": 0.0}, 1.0

    rest = 1.0 - close_p
    probs = {a: round(rest * w / total, 4) for a, w in weights.items()}
    probs["CLOSE"] = round(close_p, 4)
    # Put the rounding residue on the largest non-close share so the four sum to 1.
    top = max(weights, key=weights.get)
    probs[top] = round(probs[top] + 1.0 - sum(probs.values()), 4)

    action = max(weights, key=weights.get)
    return probs, action
