# Changelog

TradeFlow follows [Semantic Versioning](https://semver.org). While it stays on `0.x` it is an experimental side project: anything may change. The version moves only for significant updates; each gets a `vX.Y.Z` tag and a GitHub release.

## [0.1.0] - 2026-09-29

The system now finds its own stocks, argues every trade with LLM agents before it opens, keeps every position protected at the broker, and takes profit at +1R. It trades a paper account; **no edge has been demonstrated yet** (see *Status*).

### Added
- **Stock discovery** (`api/scout/`). An hourly ranking of the stocks worth watching today:
  - **Pool:** Alpaca's most-active stocks and top gainers, the last day of news, Reddit mentions (ApeWisdom), StockTwits trending, and what the world's exchanges are trading.
  - **Scoring:** past-window performance, today's move, home-exchange move, news and discussion.
  - Every ranking is logged; `python -m scout rank|scorecard|backtest` evaluates them.
- **World exchanges.** The most-traded stocks on London, Xetra, Paris, Amsterdam, Zurich, Hong Kong, NSE, Tokyo, Korea and Taiwan, mapped by company name to US lines Alpaca can trade:
  - An SEC foreign-issuer check keeps same-named US companies out.
  - One watch slot per exchange.
  - Stocks with no US line, or only an OTC one, are shown with the reason.
- **Watcher agent.** Follows each pick every 5 s. The `scout` strategy trades a pick once the watcher's confidence has held above the bar.
- **Trade desk** (`api/desk/`). No buy goes out without approval:
  - **Observer** watches the signal for 90 s.
  - **Analyst**: `qwen3.5:9b` with reasoning, entirely on the GPU.
  - **Critic**: `qwen3:4b`, split between GPU and RAM.
  - **Decision**: the mean probability must beat the trade's breakeven plus a margin set by the risk dial.
  - The case file carries the user's risk appetite. The reasoning streams live to the dashboard. Every case is logged with its trade's outcome.
- **News ingestion.** Every headline is logged with its named entities, event keywords and scores.
- **Profit-taking** (`api/engine/profit_manager.py`):
  - At +1R, half is sold and the stop moves to breakeven.
  - The stop then trails the high by 1R.
- **Broker-side protection.** Entries are real bracket orders, and a check every 60 s gives any unprotected position an OCO stop and target.
- **Spread monitor** (`api/feeds/spreads.py`). A median of the consolidated (SIP) quotes from 16 minutes earlier, falling back to a 2-minute IEX median.
- **Dashboard**:
  - *Today's watch*, sorted live by confidence.
  - A world-exchanges board.
  - The *Trade desk* and *News ingestion* cards.
  - Times in the operator's timezone (CET).
  - Bought/sold amounts and P&L on fills.
- **PPO policy** (`rl/`), research harness (`research/`) and few-shot forecasting harness (`fewshot/`). The policy learns in shadow and retrains after each close; it trades only after passing its promotion gate.

### Changed
- **US equities only**; crypto was removed.
- **Positions:** the exit policy is stop, target, strategy exit, end of day, plus the +1R scale-out. There are no adds.
- **Watchlist:** it starts empty, and no stock list is hard-coded.
- **Risk dial:** it is saved and survives restarts (default 4 until set). It drives the desk's margin and the agents' stance.
- **Deployment:** the backend container runs on the host network so it can reach Ollama, and the frontend's nginx proxies through `host.docker.internal`.

### Fixed
- **Stops at the broker.** The stop and target legs never reached the broker (`order_class` was missing).
- **Manager crash.** A watcher read with no price crashed every portfolio-manager cycle, so nothing could trade.
- **Missing prices.** Picks added before the stream connected never received prices.
- **Spread gates** read single IEX quotes (BE 7.4% against a real 0.07%) and a 1% placeholder.
- **Once per day.** The one-entry-per-day rule was lost on restart.
- **Reloaded fills** now get FIFO P&L.
- **Desk calibration.** The desk rejected nearly every trade: its prompts leaned toward no and its bar was 0.55. The analyst also anchored on the breakeven, and the critic echoed the analyst's number.

### Status
- **No edge found yet.** The performance component alone shows none over two years (t ≈ −0.7). News, discussion and desk decisions are judged from their logs going forward.
- **First live paper day:** 4 closed trades, −$96. The flat 0.8% minimum stop sits inside normal minute noise for volatile stocks; volatility-scaled stops are the next change to consider.
- **Backtester:** it does not model the scale-out yet.
- **Requirements:** Ollama with `qwen3.5:9b` and `qwen3:4b`; without them, entries are blocked.

## [0.0.0] - 2026-09-27

Initial pre-release: quant strategy library, off-process analytics, latency tracking, stair capital mode, loss recovery, backtester and fleet agents.

[0.1.0]: https://github.com/gunjanmimo/tradeflow/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/gunjanmimo/tradeflow/releases/tag/v0.0.0
