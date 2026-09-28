"""
The PPO policy as a platform strategy (rl/live.py does the work).

Entry: the policy says long at a decision bar, and the RL mode allows trading
(auto: only a policy that passed its promotion gate; live: any deployed policy).
In shadow mode the decision is logged and shown, never executed.

Exit: the policy says flat at a decision bar. The sentinel's stop, target,
stale-price and end-of-day exits apply as they did in training.
"""
from engine.strategies.base import Strategy, StrategyContext, EntryDecision, ExitDecision


class RLPolicyStrategy(Strategy):
    name = "rl_ppo"
    display_name = "RL policy (PPO)"
    description = ("A PPO agent trained on two years of one-minute bars under the platform's own "
                   "fills, stops and costs. Trades only when its policy passed the promotion gate "
                   "on unseen days (RL_MODE=auto); otherwise it runs in shadow mode.")
    source = "TradeFlow RL (rl/)"

    @staticmethod
    def _runtime():
        from rl.live import runtime
        return runtime

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        rt = self._runtime()
        if rt.mode == "off":
            return EntryDecision(False, 0.0, "RL mode is off", blocked_by="rl_off")
        if not rt.ready():
            return EntryDecision(False, 0.0, "No trained RL policy: run `python -m rl.train`",
                                 blocked_by="rl_no_policy")
        d = rt.decide(ctx.symbol, None)
        if d is None:
            return EntryDecision(False, 0.0, "RL policy: no decision yet (needs today's bars and SPY)",
                                 blocked_by="rl_no_data")
        gates = {"p_long": d.p_long, "value": d.value, "bar": d.bar, "mode": d.mode,
                 "may_trade": d.may_trade, "approved": rt.policy.approved}
        if d.action != 1:
            return EntryDecision(False, d.p_long, f"RL policy: flat (p_long {d.p_long:.2f})",
                                 blocked_by="rl_flat", gates=gates)
        if not d.may_trade:
            why = ("policy not approved by its promotion gate" if rt.mode == "auto" else f"mode {rt.mode}")
            return EntryDecision(False, d.p_long,
                                 f"RL policy would buy (p_long {d.p_long:.2f}) - shadow: {why}",
                                 blocked_by="rl_shadow", gates=gates)
        return EntryDecision(True, d.p_long, f"RL policy: long (p_long {d.p_long:.2f}, bar {d.bar})",
                             gates=gates)

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        rt = self._runtime()
        if rt.mode == "off" or not rt.ready():
            return ExitDecision(False, 0.0, 0.0, "RL policy unavailable; stop/target/close govern")
        d = rt.decide(ctx.symbol, ctx.position or {"qty": 1})
        if d is None:
            return ExitDecision(False, 0.0, 0.0, "RL policy: no decision yet")
        if d.action == 0:
            return ExitDecision(True, 1.0 - d.p_long, 1.0, f"RL policy: flat (p_long {d.p_long:.2f})")
        return ExitDecision(False, 1.0 - d.p_long, 0.0, f"RL policy: stay long (p_long {d.p_long:.2f})")
