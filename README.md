# TradeFlow: a US-stock research and paper-trading platform with a PPO agent

> [!CAUTION]
> **Use at your own risk.** TradeFlow is a personal side project, **not a battle-tested trading tool**. It can place real orders, and automated trading can lose money quickly.
>
> **Do not use it if you don't understand the computation behind it, quantitative finance, and how stock markets work.** Nothing here is financial advice. Strategies, backtests and analytics can be wrong, and past behaviour does not predict future results.
>
> Start with **paper trading** (the default), read the code before trusting it, and never trade money you cannot afford to lose. The software is provided "as is", without warranty of any kind (see [LICENSE](LICENSE)).

TradeFlow day-trades **US equities** on Alpaca. By default **news sentiment trades**: headlines scored by Jev (TypeSafe), with Laya as fallback, drive the `news_catalyst` strategy, which enters on several fresh, agreeing, bullish headlines confirmed by the price trend.

Beside it, a reinforcement-learning agent (PPO) runs in **shadow mode**. Every five minutes it decides whether each watched stock should be held or flat and logs that decision. It retrains after each close, and never places an order. It trains on one-minute bars under the same fills, stops, costs and session rules that live trading uses. It may only trade (`RL_MODE=auto`) once it has made money, with statistical significance, on days it never trained on.

The platform is built around one question: **does this have an edge after costs?** Every idea goes through the same pipeline before it can trade:

```
research/  event study of a signal ──►  backtest/  replay of the live rules ──►  rl/  PPO + walk-forward gate ──►  paper (auto)
           gross edge vs 6 bps cost                 gross vs cost per trade             beats flat, holding, random
```

---

## 📉 What the data says (read this first)

TradeFlow used to trade crypto and stocks on a stack of technical signals and exit rules. On paper it lost about $13k in two days while its own ledger reported −$1.6k. The rebuild measured why:

| | finding |
|---|---|
| Crypto | ~60 bps of fees and spread per round trip against a gross edge of ~0.7 bps per trade. **Removed.** |
| Live data | Signals ran on 0.5-second price polls with a hard-coded spread and constant volume, so the spread and volume gates never fired and no backtest matched live trading. **Fixed:** everything runs on the same one-minute bars. |
| Accounting | A close was booked at the last mark, not at the fill, so fees and slippage never reached the budget or the loss halt. **Fixed:** re-booked at broker fills; the halt also reads broker equity. |
| Exits | Selling half of any winner, delaying stops, and averaging into losers cut winners and let losers run. **Replaced** by one exit policy: stop, target, strategy exit, end of day. |
| Signals | Across 43 large caps × 499 sessions, none of nine classic intraday signals (reversal, VWAP, RSI(2), gap fade, momentum, opening-range breakout, EMA cross…) clears a 6 bps round trip. The best gross edge is +6.8 bps/trade with t = 0.8. `python -m research edge` |
| PPO | The learner works: on synthetic markets it finds a planted edge (+170 to +255 bps/day on unseen days) and learns to **stay flat** when there is none. On real bars the best policy trades 0.27 times a day per stock and loses 1.3 bps/day on unseen days, against −5.3 for random trading at that rate and −11.6 for holding. **The gate refuses it**, so it runs in shadow mode. |

This is a finding, not a failure: liquid large caps on one-minute price and volume features are close to efficient after costs. To find an edge, change the **information** (news, earnings events, order flow, less efficient names) or the **horizon** (overnight, multi-day). Changing the model won't do it. The pipeline is built to test exactly that kind of idea.

---

## 🏗️ Architecture

```
 Alpaca IEX stream ─ 1-min bars + quotes ─► minute_bars (authoritative OHLCV)   SPY for market context
                                                    │
                ┌───────────────────────────────────┼─────────────────────────────┐
                ▼                                   ▼                             ▼
   Portfolio manager (every 2s)            Sentinel (per position)          Learner (after close)
   strategy entry signal: rl_ppo           stop · target · strategy exit    download new bars
   ranked by conviction + diversification  stale price · flat by 15:50       retrain PPO (warm start)
                │                                   │                        deploy only if it beats
                ▼                                   ▼                        the current policy
   Risk guard: session, daily-loss halt (booked AND broker equity), drawdown,
   positions, sector/region caps, hard budget  ──►  Executor: bracket order (stop + target at the broker)
                                                          │
                                                 Fill reconciler: every close re-booked at the broker fill
```

- **One bar series everywhere.** Live indicators, the backtester, the research harness and the RL environment all read the same one-minute IEX bars, keyed by their start minute. Quotes and trade prints move the live price but never add an indicator sample.
- **One cost model** (`core/costs.py`): half-spread + slippage per market fill, 6 bps round trip by default. Measure your own with the live spread estimate (`state.spread_estimate`).
- **One exit policy** (`engine/sentinel_agent.py`): the stop closes on the first print through it and is also held at the broker as a bracket leg; the target is a resting limit; the strategy that opened the position may close it; everything is flat by 15:50; a position with no price for 2 minutes is closed.
- **Accounting from the broker** (`engine/fills.py`): the realized P&L of every close, including broker bracket fills, is corrected to the actual fill, so the budget, stair ladder, daily ledger and loss halt see what the account really made.

---

## 🚀 Quick start

### 1. Configure Alpaca paper keys
```bash
cp api/.env.example api/.env      # fill ALPACA_API_KEY / ALPACA_SECRET_KEY from your paper account
```
With placeholder keys TradeFlow runs a simulated feed and simulated fills.

### 2. Get data and train the policy (on the host; needs torch, a GPU helps)
```bash
cd api
pip install -r requirements-rl.txt
python -m research download --days 730     # ~7.4M one-minute bars, ~80 MB, 5 minutes
python -m research edge                    # does any simple signal have an edge after costs?
python -m rl.train --synthetic             # sanity check: PPO must find a planted edge
python -m rl.train                         # train on real data, evaluate, gate, deploy
cat models/rl/report.md
```

### 3. Run the platform
```bash
docker compose up --build                 # dashboard http://localhost:5173, API http://localhost:8000
```
The live engine runs the deployed policy in numpy; the Docker image does not need torch. Trading starts **paused**: press *Active* in the header. The **RL policy** chip shows whether the policy trades or runs in shadow mode.

---

## 🤖 The RL policy (`rl/`)

| | |
|---|---|
| Episode | one stock, one session, flat at the open and by 15:50 |
| Decisions | every 5 minutes (close of 09:34, 09:39, …, 15:44); orders fill at the next bar's open |
| Actions | flat / long. No new entries fill after 15:29 (masked) |
| Observation | the last 30 bars of 9 per-bar features (returns, range, volume, VWAP and EMA distance, RSI, SPY), 17 session and market features (time of day, gap, return since open, SPY and stock-minus-SPY returns over 5–60 min, ATR), 5 position features |
| Reward | the step's net log return of the position after costs; no shaping |
| Rules | bracket stop 1.5×ATR (0.8%–6%) and target 2R from `engine/brackets.py`, checked bar by bar, stop first, gaps at the open |
| Network | separate actor and critic MLPs (64-32, tanh). The policy starts mostly flat, so each trade has to be earned |
| Training | PPO with GAE, clipped policy and value losses, entropy bonus, advantage normalisation, linear LR decay, and a cap on how often each training session is replayed |
| Split | by date: oldest 60% train, next 20% validation (picks the checkpoint), newest 20% test (read once) |
| Gate | test mean > 0 with t ≥ 2, Sharpe above always-long, validation positive. Only then does `RL_MODE=auto` trade |
| Champion / challenger | a new policy replaces the deployed one only if it beats it on the same test days |

**Modes** (`RL_MODE`, or the RL panel): `shadow` (default) decides and logs but never trades, `auto` trades only an approved policy, `live` trades any deployed policy (paper only, please), `off` disables it. While another strategy trades, the portfolio manager still asks the policy for a shadow decision on every watched stock. To let the policy trade in its own right, make it the default strategy (`POST /api/strategies/class` with `"strategy":"rl_ppo"`) and set the mode to `auto` or `live`. Every live decision is appended to `api/datasets/rl/live_decisions.jsonl`.

**Proof it learns.** `tests/test_rl.py` trains PPO on synthetic markets. It must earn well over +50 bps/day on unseen days when a regime edge is planted, and must stop trading when there is none. Other tests pin the environment's accounting to hand-computed trades, and hold live features equal to the training features.

```bash
curl localhost:8000/api/rl                       # mode, gate, test results vs baselines, learning curve
curl -X POST localhost:8000/api/rl/mode -H 'content-type: application/json' -d '{"mode":"shadow"}'
curl -X POST localhost:8000/api/rl/retrain       # download new bars + warm-start retrain now
```

---

## 🔬 Research and backtesting

```bash
python -m research edge --signals reversal_z,gap_fade --horizons 15,60,120
python -m backtest --days 30 --symbols NVDA,AAPL --strategies supertrend,zscore_reversion
```

- **`research edge`**: an event study per signal and horizon. Entry at the next open, no overlapping trades, no overnight holds, excess return over a random entry, t-statistics clustered by day, net of costs, with a chronological 40% holdout. Tests assert that no signal looks ahead.
- **`backtest`**: replays any platform strategy through the live exit rules and reports gross edge per trade against costs per trade.
- The classic strategy library (`engine/strategies/library.py`: Bollinger, Connors RSI(2), z-score, VWAP, MACD, Supertrend, Donchian and others) is still available per symbol or as the default (`POST /api/strategies/class`). None of them shows an edge after costs on this data.

---

## 🪜 Capital Mode: Classic vs Stair

The **Capital mode** chip in the header switches between two ways of managing money. It takes effect on the next evaluation.

- **Classic:** the bots trade a fixed budget.
- **Stair (profit ratchet):** the deposit is split into *trading capital* and a *reserve* the engine never trades. Each step has a target, by default 2× the step's starting capital. When closed trades reach it, a share of that step's profit (default 50%) is **banked as income** and removed from the trading budget. The rest compounds into the next step.

| Step | Trade with | Target (2×) | Bank 50% | Total banked |
|---|---|---|---|---|
| 1 | $500 | $1,000 | $250 | $250 |
| 2 | $750 | $1,500 | $375 | $625 |
| 3 | $1,125 | $2,250 | $562 | $1,187 |

(Example: $1,000 deposit, 50% traded, $500 reserve untouched.)

Rules:
- **Progress counts realised PnL only.** Unrealised gains are never banked.
- **Stair never raises risk to reach a target.** Strategies, stops and the risk dial are unchanged.
- **Main broker equity is locked.** When a budget or deposit is assigned to the bot in classic or stair mode, the main broker equity outside the budget cap is strictly locked (`locked_broker_equity = max(0, broker_equity - assigned_capital)`). The bot operates exclusively within its hard cap and never touches locked broker equity or cash.
- **If trading capital falls below the smallest order the engine can place, new entries stop.** The reserve and banked income are never used to top it up.
- **Stair refuses to start with capital too small to trade.** At risk dial 4, one position is capped at 15% of capital, so it needs at least $200 of trading capital.
- **The engine cannot move money.** Reserve and banked income stay as cash at the broker; withdraw them there.
- **The ladder is saved to `api/data/capital_plan.json`,** so it survives restarts. Banked income carries over when a ladder is restarted.

```bash
curl localhost:8000/api/capital-plan
curl -X POST localhost:8000/api/capital-plan -H 'content-type: application/json' \
     -d '{"mode":"stair","deposit":1000,"deploy_pct":0.5,"target_multiple":2,"harvest_pct":0.5}'
curl -X POST localhost:8000/api/capital-plan -d '{"mode":"classic"}' -H 'content-type: application/json'
```

---

## 🧮 Daily Profit & Loss Calculator

TradeFlow includes a built-in **Daily Profit & Loss Calculator & Expectancy Simulator**:
- **Live Performance & Accounting:** Real-time breakdown of today's realized PnL, open unrealized PnL, net PnL, return % on bot budget, win rate %, and profit factor.
- **Position Scenarios:** Evaluates potential outcomes if all active positions hit Take Profit (best case) vs Stop Loss (worst case).
- **Interactive Expectancy Simulator:** Allows simulating expected value (EV) per trade, breakeven win rate, and projected daily PnL based on customizable target profit, max daily loss limit, win rate %, and reward-to-risk ratio.
- **30-Day Historical Ledger:** Persisted daily performance records in `api/data/pnl_ledger.json`.

```bash
# Get today's PnL breakdown, 30-day history, and scenarios
curl localhost:8000/api/daily-pnl

# Run projection simulations
curl -X POST localhost:8000/api/daily-pnl/calculator -H 'content-type: application/json' \
     -d '{"target_daily_profit": 100, "max_daily_loss": 50, "planned_trades": 5, "win_rate_pct": 60, "reward_risk_ratio": 2.0, "risk_per_trade": 25}'
```

---

## 🌍 Discovery & Diversification

Discovery finds stocks you don't hold yet. Diversification keeps the book from being one bet. Both are driven by the same 1-10 risk dial.

**Candidate pool** (`engine/discovery.py`). Candidates are kept separate from the watchlist and come from:
- SEC Form 4 insider trades and eToro top-investor holdings, for any symbol, not only watched ones.
- Our own universe screen: GICS sector leaders, US-listed ADRs of UK, European, Chinese, Japanese and Indian companies, and sector and country ETFs.

Each candidate is scored on four components:

| Component | Weight | Inputs |
|---|---|---|
| Smart money | 30% | Insider buys and sells, copy-trader holdings |
| Public sentiment | 20% | Alpaca news scored by Laya, StockTwits bull/bear ratio |
| Momentum | 30% | 12-1m momentum, 3m strength vs its sector ETF, 200-day trend |
| Diversification fit | 20% | Sleeve headroom, under-target regions, correlation with holdings |

Missing components don't vote. Nothing is traded until you promote a candidate to the watchlist.

**Classification** (`core/universe.py`). Every symbol gets a GICS sector, theme, country, region and currency. Unknown US tickers are classified from their SEC SIC code. Foreign listings (`.L`, `.NS`, `.HK`...) map to their US ADR when one exists. Otherwise they stay visible but untradable, with a country ETF offered as the proxy. Alpaca only trades US listings, so Indian energy and defence names are reachable only through INDA, EPI or SMIN.

**Entry gate** (`engine/diversification.py`). Every buy is clamped to the headroom in its sleeves:

| Dial | Sector | US | Europe / Asia | Min defensive | Correlation halving at |
|---|---|---|---|---|---|
| 1 | 20% | 50% | 20% | 30% | 0.60 |
| 4 (default) | 30% | 60% | 20% | 20% | 0.75 |
| 10 | 50% | 90% | 40% | 0% | 0.90 |

- Percentages are of the trading budget. Levels 2, 3 and 5–9 are in `core/risk_profile.py`.
- A new position more correlated than the limit with an existing holding gets half size.
- An entry with no headroom is refused.

**Risk analysis.** Computed from about 90 days of daily returns: portfolio volatility, parametric and historical 1-day VaR/CVaR, diversification ratio, effective number of bets, and each sector's and region's share of total risk. The what-if view shows how a standard-size position in any candidate would change these numbers.

Open it from the **Discovery** chip in the dashboard header. API: `/api/discovery`, `/api/discovery/promote`, `/api/discovery/what-if/{symbol}`, `/api/diversification`.

---

## 🏷️ Versioning

Current version: **`0.0.0`** (pre-release).

TradeFlow follows [Semantic Versioning](https://semver.org). While it stays on `0.0.x`, treat it as an experimental side project: anything may change. The version is **not bumped for every change**. It moves only when a significant update lands, and each bump gets a matching git tag (`vX.Y.Z`) and GitHub release.

The version lives in `api/core/version.py` (backend, also reported by `GET /api/status`) and `app/package.json` (frontend). Keep them in sync when bumping.

---

## 🤝 Contributing

TradeFlow is a **free, open-source, AI-based trading platform for your own personal use**, and pull requests to improve it are very welcome: bug fixes, new strategies, better risk controls, tests, documentation or UI work.

1. Fork the repo and create a branch from `main`.
2. Keep changes focused, and explain the *why* in the PR description. For trading logic, include how you tested it.
3. Never commit secrets: `api/.env` and broker keys stay local.
4. Only contribute code you wrote or that carries an MIT-compatible license. Please do not paste in code from GPL/AGPL projects.

Found a problem but don't have a fix? Opening an issue helps too.

## 📄 License

Released under the [MIT License](LICENSE). Third-party services, models and data sources this project connects to (Alpaca, the Laya model, SEC EDGAR, eToro, StockTwits and others) are governed by their own terms, which you are responsible for following.
