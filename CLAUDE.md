# TradeFlow

A US-stock day-trading engine on an Alpaca **paper** account, built by one person to learn quant finance. The goal is a system that makes money *measurably*. **No edge has been demonstrated yet** (see `CHANGELOG.md` → Status). Report results honestly, losses included; never present an untested idea as an improvement.

**Stack:** FastAPI backend (`api/`, Python 3.12), React + Vite dashboard (`app/`), Docker Compose. Local LLMs through Ollama; Alpaca for data and orders.

## Where things are

- `README.md`: architecture, findings, every subsystem. `CHANGELOG.md`: what changed per version.
- Pipeline:
  1. `api/scout/`: hourly discovery and a live watcher
  2. `api/engine/portfolio_manager.py`: ranks entries
  3. `api/desk/`: LLM trade desk that must approve every buy
  4. `api/engine/executor.py`: orders
  5. `api/engine/sentinel_agent.py` and `api/engine/profit_manager.py`: exits
- Research and learning: `api/research/` (edge after costs), `api/backtest/`, `api/rl/` (PPO, shadow), `api/fewshot/`.
- Settings: `api/core/config.py`. Every setting is documented there; read it before adding one.
- Each module opens with a docstring stating its rules and why. Read it before changing the module.

## Verify every change

```bash
cd api && CUDA_VISIBLE_DEVICES="" python -m pytest -q      # all tests; they must pass
cd app && npx vite build                                   # frontend must build
```
- `api/.venv` is broken. Use a Python with `api/requirements.txt` (+ `requirements-rl.txt` for `rl/`).
- `CUDA_VISIBLE_DEVICES=""`: Ollama holds most of the 8 GB GPU, so the torch tests OOM on it.
- Tests must never touch `api/data/` or the live account. `api/tests/conftest.py` redirects every persisted file; extend it when you add one.
- For a live check without disturbing the running engine, start a second backend with `ALPACA_API_KEY=PK_PLACEHOLDER_KEY` on port 8001 and use `VITE_API_BASE=http://localhost:8001 npx vite`. Alpaca allows **one** data websocket per account.
- CLIs: `python -m scout rank|scorecard|backtest`, `python -m research edge`, `python -m backtest --symbols X,Y`, `python -m rl.train`.

## Running system: gotchas that have bitten

- **IMPORTANT: every backend start leaves trading PAUSED** (a safety default). After any restart you cause, re-enable it with `POST /api/toggle-trading` if it was on, and say so.
- Deploy with `docker compose restart backend`. The container mounts `./api`, but uvicorn does not reload, so code changes take effect only after a restart. Check `GET /api/positions` first.
- **Networking:**
  - The backend runs with `network_mode: host`: it reaches Ollama at `localhost:11434`, and a test backend can't use port 8000.
  - The frontend is a built nginx image that proxies to `host.docker.internal:8000`. Rebuild it with `docker compose up -d --build --no-deps frontend`.
- `api/data/` is root-owned (written by the container) and gitignored.
- The desk needs Ollama with `qwen3.5:9b` (analyst) and `qwen3:4b` (critic). If it is unreachable, **entries are blocked**, not bypassed.
- **GPU:** both models share the 8 GB GPU: the analyst fully on it, the critic split GPU+RAM (`DESK_CRITIC_NUM_GPU`).
  - Swapping models costs minutes.
  - Changing `num_ctx` reloads a model.
  - Benchmarks against Ollama during market hours slow the live desk.
- Live quotes are IEX-only and overstate spreads badly. Spread gates must read `feeds/spreads.py` (delayed-SIP median), never a single quote.
- A fresh stream subscription has no price until its first minute bar closes. Don't read "no price" in the first minute as a bug.
- The risk dial comes from the dashboard and is saved (`data/risk_dial.json`, default 4). Read it live; never assume a value.

## Design rules

- **No hard-coded stocks to watch or trade.** The watchlist starts empty; everything is discovered. The curated table in `core/universe.py` is sector/country metadata only.
- **One data path, one exit policy.** Live, backtest and RL use the same one-minute bars and cost model.
  - Exits: stop, target, strategy exit, end of day, plus the +1R scale-out.
  - No adds.
  - Don't reintroduce the removed exit stack (any-profit harvest, averaging into losers). The README says why.
- **Judge a new strategy before it trades:** `research edge` → `backtest` → the RL gate. The PPO policy stays in shadow until it passes its gate.
- **Fail closed.** A missing or failed source means *unknown* or *no trade*, never a default that trades.
- **The desk:**
  - The numbers decide, not the models' labels.
  - The analyst is never shown the breakeven, and the critic never sees the analyst's probability (small models anchor on and echo them).
- **Broker state beats engine state.** Sell quantities come from the broker's live position. Every position needs a broker-side stop (bracket or OCO).
- **Times shown to the user are Europe/Paris (CET/CEST).** Market logic is New York time.
- Long-only US equities (including ADRs). Crypto was removed on purpose; don't bring it back.

## Git and releases

- Branch from `main`, commit with a descriptive body, open a PR with `gh`.
- `main` has a "Protect main" ruleset: one approving code-owner review, no direct pushes. The owner is the only maintainer, so merge with `gh pr merge N --merge --admin`, **only after the user says so for that PR.**
- A release bumps `api/core/version.py`, `app/package.json` (+ its lockfile) and the README version line. It adds a `CHANGELOG.md` entry, an annotated `vX.Y.Z` tag on the merge commit, and a GitHub release.
- Never commit `api/.env`, `api/data/`, `api/datasets/` or `api/models/`.

## When compacting

Keep: the files modified, the test and build commands, whether the backend was restarted and trading re-enabled, open positions, and any decision still waiting on the user.
