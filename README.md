# TradeFlow: Sub-Second Quant & Laya Trading Terminal

> [!CAUTION]
> **Use at your own risk.** TradeFlow is a personal side project, **not a battle-tested trading tool**. It can place real orders, and automated trading can lose money quickly.
>
> **Do not use it if you don't understand the computation behind it, quantitative finance, and how stock and crypto markets work.** Nothing here is financial advice. Strategies, backtests and analytics can be wrong, and past behaviour does not predict future results.
>
> Start with **paper trading** (the default), read the code before trusting it, and never trade money you cannot afford to lose. The software is provided "as is", without warranty of any kind (see [LICENSE](LICENSE)).

An autonomous, continuous quantitative trading system combining **Convai Innovations' Laya decision model** for sub-second sentiment scoring, a high-frequency **Quant Matrix ($t_1 \dots t_N$)**, deterministic risk management, and the **Alpaca Trading API**.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               4-PILLAR MARKET INTELLIGENCE CONSENSUS                   │
│  1. SEC Form 4 (Insider Purchases via SEC EDGAR)       [Weight: 35%]   │
│  2. eToro (Top Sharpe Popular Investor Portfolios)     [Weight: 25%]   │
│  3. Dub.app / Public.com (Thematic Creator Pies)       [Weight: 20%]   │
│  4. StockTwits (Sentiment Leaders & Bull/Bear Ratios)  [Weight: 20%]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         (Consensus Score & Thesis)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Laya Decision Model (Convai 421M ModernBERT)                        │
│    Fast in-process probability scoring: { pos_prob, neg_prob, open }   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. In-Memory Quant Matrix (t1...tN)                                    │
│    Sub-millisecond indicators: EMA 9/21, RSI 14, ATR, Spread, Volume  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Deterministic Risk Guard & Alpaca Paper Execution                   │
│    2% equity risk sizing, automated Bracket Orders (SL + TP)           │
└────────────────────────────────────────────────────────────────────────┘
```

### The Sub-Second Execution Path
1. **In-Process Laya Model**: Evaluates news headlines and catalyst statements into calibrated probabilities (`pos_prob`, `neg_prob`) in **15–35ms** without external LLM API roundtrips.
2. **Asynchronous Memory Cache**: Sentiment scores are pre-cached in RAM (`0.001ms` lookup).
3. **C-Accelerated Quant Matrix**: RSI, EMA, and volatility are computed on rolling circular buffers in **< 0.05ms**.
4. **Alpaca Execution**: Automated bracket orders (Entry + Stop Loss + Take Profit) sent via `alpaca-py`.

---

## 🚀 Quick Start

### 1. Configure Alpaca Paper Trading Keys
Create or edit `api/.env`:
```bash
cp api/.env.example api/.env
```
Fill in your keys from your [Alpaca Paper Account](https://alpaca.markets):
```env
ALPACA_API_KEY=PK_YOUR_PAPER_KEY
ALPACA_SECRET_KEY=YOUR_PAPER_SECRET
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
IS_PAPER=True
```
*(Note: If left as placeholders, TradeFlow automatically runs in **simulated paper trading mode** so you can test the entire pipeline safely without an account).*

### 2. Launch with Docker Compose
Run both backend and frontend in containers with a single command from the root directory:
```bash
docker compose up --build
```
- **Web Dashboard**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Backend & Telemetry**: [http://localhost:8000](http://localhost:8000) (WebSocket at `ws://localhost:8000/ws`)

To run in detached/background mode:
```bash
docker compose up -d
```
To stop the containers:
```bash
docker compose down
```

---

## 📊 Features & Asset Classes

- **Dual Asset Support (US Equities + 24/7 Crypto)**:
  - **US Equities**: `NVDA`, `AAPL`, `MSFT`, `PLTR`, etc. (NYSE / NASDAQ).
  - **24/7 Cryptocurrencies**: `BTC/USD`, `ETH/USD`, `SOL/USD` (Trades continuously 24/7/365, including weekends and evenings!).
  - **Fractional Sizing for Crypto**: Automatically sizes positions in decimal quantities (e.g. `0.024 BTC`).
- **4-Pillar Consensus Engine**: Fuses **SEC Form 4 / On-Chain Whale Flows** (35%), **eToro Copy Portfolios** (25%), **Dub/Public Thematic Pies** (20%), and **StockTwits Sentiment** (20%).
- **In-Process Laya Model**: Classifies financial catalysts and news headlines into calibrated probabilities in **15–35ms**.
- **C-Speed Quant Matrix ($t_1 \dots t_N$)**: Evaluates EMA 9/21, RSI 14, ATR volatility, and bid-ask spreads in **< 0.05ms**.
- **Interactive Dashboard**: Filter by `ALL`, `STOCKS`, or `CRYPTO (24/7)`, inject live news to test Laya, and manage positions.
- **Emergency Kill Switch**: Panic button to liquidate open positions and halt trading immediately.

---

## 🧠 Quant Strategy Library

`api/engine/strategies/library.py` holds fifteen price-driven strategies. They are re-implemented from the published methods, not copied: backtrader and freqtrade are GPL-3, and [FinceptTerminal](https://github.com/Fincept-Corporation/FinceptTerminal) is AGPL-3.0, so copying their code would relicense this project. [Krexibd/quant-trading](https://github.com/Krexibd/quant-trading) is MIT and is credited on each strategy that came from it.

| Strategy | Family | Built for regime | Source |
|---|---|---|---|
| `bollinger_reversion` | Mean reversion | ranging, mixed | Bollinger |
| `connors_rsi2` | Mean reversion | ranging, trending_up, mixed | Connors & Alvarez |
| `zscore_reversion` | Stat. reversion | ranging | Chan (O-U process) |
| `vwap_reversion` | Mean reversion | ranging, mixed | VWAP benchmark |
| `macd_trend` | Trend | trending_up, mixed | Appel |
| `ma_crossover` | Trend | trending_up | zipline/backtrader classic |
| `donchian_turtle` | Breakout | trending_up, volatile | Turtle rules |
| `supertrend` | Trend | trending_up, volatile | Seban |
| `ts_momentum` | Momentum | trending_up | Moskowitz, Ooi & Pedersen (2012) |
| `volatility_squeeze` | Volatility | volatile, mixed, ranging | Bollinger "The Squeeze" |
| `parabolic_sar` | Trend | trending_up, volatile | Wilder, via quant-trading |
| `awesome_saucer` | Momentum | trending_up, mixed | Bill Williams, via quant-trading |
| `heikin_ashi` | Reversal | trending_up, mixed, volatile | Heikin-Ashi, via quant-trading |
| `candle_reversal` | Reversal | ranging, mixed | Hammer / shooting star, via quant-trading |
| `pairs_reversion` | Stat. arb | all except trending_down | Engle-Granger pairs, via quant-trading |

Lookbacks are measured in samples of the engine's price buffer (up to 250), not days. Candle strategies build OHLC micro-bars from groups of 5 samples.

### Zero-latency architecture

Everything heavier than a dictionary lookup runs in a **separate analysis process** (`engine/analysis/`, spawned once, every `ANALYSIS_INTERVAL_SECONDS`). A thread would still contend for Python's GIL with the tick path; a process does not. The worker computes:

- **Regime** per symbol (`regime.py`): efficiency ratio, EMA slope, vol ratio, Hurst exponent.
- **Quant council** (`council.py`): every library strategy votes, weighted by regime fit and realised track record.
- **Monte Carlo**: block-bootstrap probability that the take-profit is hit before the stop, with and without recent drift.
- **Pairs**: the best Engle-Granger cointegrated partner per symbol, pre-filtered by return correlation.
- **Portfolio risk**, following FinceptTerminal's ideas (no code): ledger Sharpe, Sortino, profit factor, expectancy and max drawdown; variance-covariance VaR and CVaR of open positions, with correlations.

The hot path only **reads** these results:

- **Manager (executor):** refuses an entry when the council opposes it, or when Monte Carlo P(TP first) is below `MC_MIN_TP_FIRST_PROB` (0 = off).
- **Trade bots (sentinels):** exit on a decisive bearish council read. A position's exits always follow its `entry_strategy`.
- **Adaptive strategy:** uses the worker's regime and ranked shortlist, and re-checks only the top `ADAPTIVE_TOP_K` against the live tick.

Missing or stale analysis means "no opinion". It is never computed inline as a fallback.

The hot path itself was also sped up: EMA and RSI moved from Python loops to a C-level IIR filter (identical results to 1e-15), and the rolling std is now O(n). Same harness, original commit vs. now:

| Tick → decision | p50 before | p50 after |
|---|---|---|
| Default strategy | 190 µs | 83 µs |
| `adaptive` (opt-in) | — | 185 µs |

### Latency on the dashboard

The **Latency** chip in the header shows tick-to-decision p50/p95 and the dashboard round-trip. Click it for every stage, grouped by kind: hot path, broker round-trip, market-data age, event-loop lag and background worker. Each stage shows p50/p95/p99/max against a budget. Recording a sample costs about 0.2 µs. `GET /api/latency` returns the same data.

**API**
```bash
curl localhost:8000/api/latency                # per-stage latency percentiles
curl localhost:8000/api/quant/library          # strategies + track record
curl localhost:8000/api/quant/regimes          # live regime per symbol
curl localhost:8000/api/quant/analyze/BTC/USD  # council votes, Monte Carlo, pair
curl localhost:8000/api/quant/portfolio        # VaR/CVaR, Sharpe/Sortino, drawdown
# Let crypto bots pick strategies per moment:
curl -X POST localhost:8000/api/strategies/class -H 'content-type: application/json' \
     -d '{"asset_class":"crypto","strategy":"adaptive"}'
```

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
- **Stair refuses to start with capital too small to trade.** At risk dial 4, one position is capped at 15% of capital, so crypto needs at least $100 of trading capital (stocks $200).
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

| Dial | Sector | Crypto | US | Europe / Asia | Min defensive | Correlation halving at |
|---|---|---|---|---|---|---|
| 1 | 20% | 5% | 50% | 20% | 30% | 0.60 |
| 4 (default) | 30% | 15% | 60% | 20% | 20% | 0.75 |
| 10 | 50% | 35% | 90% | 40% | 0% | 0.90 |

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
