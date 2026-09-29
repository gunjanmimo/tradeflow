import os
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Alpaca API Credentials
    ALPACA_API_KEY: str = Field(default=os.getenv("ALPACA_API_KEY", "PK_PLACEHOLDER_KEY"))
    ALPACA_SECRET_KEY: str = Field(default=os.getenv("ALPACA_SECRET_KEY", "SECRET_PLACEHOLDER_KEY"))
    ALPACA_BASE_URL: str = Field(default=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    ALPACA_DATA_URL: str = Field(default=os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"))
    IS_PAPER: bool = True

    # Sub-second Quant Engine Settings
    MAX_CONCURRENT_POSITIONS: int = 5
    RISK_PER_TRADE_PERCENT: float = 1.0   # % of allocated budget risked per trade (stop-to-entry)
    STOP_LOSS_ATR_MULTIPLE: float = 1.5   # Stop loss at 1.5 * ATR
    TAKE_PROFIT_ATR_MULTIPLE: float = 3.0 # Take profit at 3.0 * ATR (1:2 Risk/Reward)

    # Stop-distance bounds, as a fraction of price. Relative, never absolute dollar
    # amounts, so a $5 stock and a $900 stock get comparable stops.
    MIN_STOP_DISTANCE_PCT: float = 0.008  # never risk a stop tighter than 0.8%
    MAX_STOP_DISTANCE_PCT: float = 0.06   # never accept a stop wider than 6%
    MAX_POSITION_NOTIONAL_PCT: float = 0.15  # max 15% of budget in one position
    # Headroom held back when sizing a market order, so a fill above the quoted
    # price cannot push committed capital past the hard budget cap.
    BUDGET_FILL_BUFFER_PCT: float = 0.5

    # Portfolio-level circuit breakers (evaluated before every new entry)
    MAX_DAILY_LOSS_PERCENT: float = 3.0   # halt new entries after -3% on the day
    # Also measure the day's loss on the broker account itself (equity vs. the
    # previous close, relative to the bots' budget). Booked P&L missed fees and
    # slippage and under-reported a 13% account loss as 1.6%; this cannot.
    # Turn off only if the account is also traded by hand.
    RISK_BROKER_EQUITY_GUARD: bool = True
    MAX_DRAWDOWN_PERCENT: float = 10.0    # halt new entries after -10% from peak equity

    # Signal freshness: a sentiment score older than this is not tradeable.
    # Without a TTL a score from hours ago is read as current on every tick.
    SENTIMENT_MAX_AGE_SECONDS: float = 1800.0  # 30 minutes

    # A single headline is a rumour. Require several, mostly agreeing, before a
    # sentiment reading is allowed to authorise an entry. This is the practical
    # answer to Laya shipping over-confident: trust agreement, not one probability.
    MIN_HEADLINES_FOR_ENTRY: int = 2
    MIN_HEADLINE_AGREEMENT: float = 0.60

    # Real news ingestion (Alpaca News API -- uses the existing Alpaca keys)
    NEWS_POLL_SECONDS: float = 60.0
    NEWS_LOOKBACK_HOURS: int = 12
    NEWS_BATCH_SYMBOLS: int = 20   # symbols per Alpaca news request

    # --- Real conviction sources (all fail closed when unavailable) ---
    # SEC requires a descriptive User-Agent with contact info, or it returns 403.
    SEC_USER_AGENT: str = Field(default=os.getenv(
        "SEC_USER_AGENT", "TradeFlow Research admin@tradeflow.local"))
    SEC_MAX_FILINGS: int = 25          # Form 4 filings parsed per cycle
    STOCKTWITS_MAX_SYMBOLS: int = 8
    STOCKTWITS_MIN_MESSAGES: int = 10  # ignore thin chatter
    AGGREGATOR_INTERVAL_SECONDS: float = 900.0
    CONSENSUS_MAX_AGE_SECONDS: float = 3600.0

    # eToro public API. Auth is the non-interactive credential PAIR:
    #   x-api-key   = public application key
    #   x-user-key  = user key (identifies the acting account)
    #   X-Request-Id = a fresh UUID per request (required; 422 without it)
    # Bearer tokens are mutually exclusive with this pair. Verified against the
    # live API: the pair returns 200, a bearer token returns 404/403.
    ETORO_API_KEY: str = Field(default=os.getenv("ETORO_API_KEY", ""))
    ETORO_PRIVATE_KEY: str = Field(default=os.getenv("ETORO_PRIVATE_KEY", ""))
    ETORO_CLIENT_ID: str = Field(default=os.getenv("ETORO_CLIENT_ID", ""))
    ETORO_API_BASE: str = Field(default=os.getenv(
        "ETORO_API_BASE", "https://public-api.etoro.com"))
    # Rate limit is 60 req/60s. One rankings call plus one holdings call per
    # investor, every AGGREGATOR_INTERVAL_SECONDS, must stay under that.
    ETORO_TOP_INVESTORS: int = 15
    ETORO_RANKING_PERIOD: str = "CurrYear"
    
    # Sentiment backend. "jev" = TypeSafe's hosted Jev model first, falling
    # back to in-process Laya once Jev errors out or hits its usage limit.
    # "laya" = Laya only, Jev never called.
    SENTIMENT_BACKEND: str = Field(default=os.getenv("SENTIMENT_BACKEND", "jev"))
    TYPESAFE_API_KEY: str = Field(default=os.getenv("TYPESAFE_API_KEY", ""))
    TYPESAFE_BASE_URL: str = Field(default=os.getenv(
        "TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone"))
    TYPESAFE_MODEL: str = Field(default=os.getenv("TYPESAFE_MODEL", "jev-latest"))
    TYPESAFE_TIMEOUT_SECONDS: float = 5.0
    # Transient Jev failures (timeouts, 5xx) in a row before switching to Laya
    # for good. Quota/auth errors (401/402/403/429) switch immediately.
    TYPESAFE_MAX_CONSECUTIVE_FAILURES: int = 3

    # Sentiment & Trigger Thresholds
    MIN_BUY_SENTIMENT_POS: float = 0.65  # Laya pos_prob >= 0.65
    MAX_SELL_SENTIMENT_NEG: float = 0.65 # Laya neg_prob >= 0.65 to exit early
    RSI_OVERSOLD_THRESHOLD: float = 35.0
    RSI_OVERBOUGHT_THRESHOLD: float = 70.0
    
    # --- Agent memory (HydraDB) ---
    # Minimum trades in a setup bucket before its statistics are allowed to
    # influence anything. Below this, absence of data must not read as a verdict.
    MEMORY_MIN_SAMPLES: int = 8
    # Average R at or below which a setup is actively avoided.
    MEMORY_AVOID_EXPECTANCY: float = -0.25
    MEMORY_ENABLED: bool = True

    # --- Quant council (engine/strategies/council.py) ---
    # Information only: the council's verdict is shown on the dashboard but no
    # longer vetoes entries or closes positions (it never showed an edge).
    COUNCIL_MIN_VOTERS: int = 3
    COUNCIL_SUPPORT_CONSENSUS: float = 0.20
    COUNCIL_VETO_CONSENSUS: float = -0.25   # verdict "oppose" at or below this

    # --- Off-process analysis worker (engine/analysis) ---
    # Everything heavier than a dict lookup runs in a separate process on this
    # cadence, so it can never add latency to the tick path.
    ANALYSIS_ENABLED: bool = True
    ANALYSIS_INTERVAL_SECONDS: float = 1.0
    # Results older than this are ignored by the hot path (treated as absent).
    ANALYSIS_MAX_AGE_SECONDS: float = 10.0
    # Adaptive strategy: how many worker-ranked candidates to evaluate per tick.
    ADAPTIVE_TOP_K: int = 3
    # Monte Carlo P(take-profit before stop), shown on the dashboard.
    MC_PATHS: int = 1000

    # --- Discovery & diversification (engine/discovery.py, engine/diversification.py) ---
    DISCOVERY_ENABLED: bool = True
    # Cycle cadence. Cheap when nothing changed: daily bars are cached for hours
    # and SEC lookups remember failures, so most cycles are pure in-memory scoring.
    DISCOVERY_INTERVAL_SECONDS: float = 60.0
    # Daily history used for correlation, portfolio volatility and momentum.
    DAILY_BARS_LOOKBACK_DAYS: int = 420
    CORRELATION_WINDOW_DAYS: int = 90
    # Top-ranked tradable candidates that also get public sentiment tracked
    # (Alpaca news scored by Laya, plus StockTwits), without being traded.
    DISCOVERY_SENTIMENT_TOP_N: int = 15
    STOCKTWITS_MAX_CANDIDATES: int = 8
    # Candidates not seen by any source for this long are dropped from the pool.
    DISCOVERY_STALE_SECONDS: float = 7 * 86400.0
    DISCOVERY_MAX_CANDIDATES: int = 300
    # Auto-selection: each cycle the best-scoring stocks are put on the watchlist
    # (and auto-picked ones that fall well out of the ranking are taken off, unless
    # held). Entries still pass the strategy, risk guard and diversification gates.
    # Off while the scout (scout/) picks the watchlist: two rankers adding and
    # dropping symbols would fight. The pool still scores for manual picks.
    DISCOVERY_AUTO_PROMOTE: bool = False
    DISCOVERY_AUTO_TOP_N: int = 10
    DISCOVERY_AUTO_MIN_SCORE: float = 0.55
    # Extra auto-pick slots for stocks the smart-money sources are clearly bullish
    # on (SEC insider buying, eToro top-investor holdings), provided our own score
    # does not contradict them.
    DISCOVERY_AUTO_SMART_MONEY_N: int = 5
    DISCOVERY_AUTO_SMART_MONEY_MIN_CONSENSUS: float = 0.70
    DISCOVERY_AUTO_SMART_MONEY_MIN_SCORE: float = 0.50

    # --- Stock score strategy (engine/strategies/stock_score.py) ---
    # Our own per-stock score: daily momentum + intraday structure always vote;
    # smart money and news vote only when they have real data.
    STOCK_SCORE_MIN_ENTRY: float = 0.62
    STOCK_SCORE_EXIT_BELOW: float = 0.40
    # No discretionary (score/news) exit before this; stops and targets still apply.
    STOCK_SCORE_MIN_HOLD_MINUTES: float = 30.0

    # --- US pre-market (04:00-09:30 NY) ---
    # Alpaca accepts only DAY limit orders flagged extended_hours then, and no
    # brackets: stop and target are enforced by the position's sentinel instead.
    # Off by default: every backtest and the RL policy's training data are
    # regular-session bars, so a pre-market entry is outside anything tested.
    PREMARKET_TRADING_ENABLED: bool = False
    PREMARKET_MAX_SPREAD_PCT: float = 1.0      # refuse entries on wider quotes
    PREMARKET_LIMIT_OFFSET_PCT: float = 0.10   # buy limit this far above the ask
    PREMARKET_EXIT_OFFSET_PCT: float = 0.50    # sell limit this far below the bid
    # An unfilled extended-hours exit is left working this long, then re-priced.
    # No second exit is sent meanwhile: that oversold longs into shorts.
    EXTENDED_EXIT_REPRICE_SECONDS: float = 30.0
    # Each re-price sits this much further through the touch, up to the cap. A
    # quiet pre-market gives no fresh quote, so re-pricing off the same last price
    # re-sent the identical limit every cycle and the exit never filled.
    EXTENDED_EXIT_STEP_PCT: float = 0.50
    EXTENDED_EXIT_MAX_OFFSET_PCT: float = 3.0
    PREMARKET_SIZE_FACTOR: float = 0.5         # half size: thin, jumpy market
    PREMARKET_STALE_ORDER_SECONDS: float = 120.0
    PREMARKET_UNFILLED_COOLDOWN_SECONDS: float = 900.0
    PREMARKET_QUOTE_POLL_SECONDS: float = 15.0
    # A polled quote older or wider than this is not a price and is ignored.
    PREMARKET_MAX_QUOTE_AGE_SECONDS: float = 300.0
    PREMARKET_MAX_QUOTE_SPREAD_PCT: float = 2.0
    # Held stocks stream every trade print. If none arrives for this long, the
    # position is priced from the broker's own mark, refreshed every
    # POSITION_MARK_POLL_SECONDS, so its stop and target keep being checked.
    POSITION_MAX_PRICE_AGE_SECONDS: float = 5.0
    POSITION_MARK_POLL_SECONDS: float = 1.0
    # Seconds a sized-but-unfilled buy keeps its notional reserved against the
    # sector/region caps, so two concurrent entries cannot both use one headroom.
    DIVERSIFICATION_RESERVATION_SECONDS: float = 180.0

    # --- Forced exits (engine/forced_exits.py) ---
    # A held position whose price has not changed for this long is dead capital
    # with an unpriced risk. It is closed at once -- unless it is already down by
    # STALE_PRICE_MAX_LOSS_PCT or more, where dumping it would lock in a big loss
    # and the stop / other exits keep ownership.
    STALE_PRICE_EXIT_SECONDS: float = 120.0
    STALE_PRICE_MAX_LOSS_PCT: float = 5.0
    # A position worth less than this is a leftover (the engine buys whole shares
    # and never smaller than engine/risk_guard.MIN_ORDER_DOLLARS): close it.
    DUST_POSITION_USD: float = 30.0
    # A forced exit never sits waiting longer than this: a failed close is retried
    # within it, whatever the normal backoff would have been.
    FORCED_EXIT_MAX_WAIT_SECONDS: float = 50.0

    # --- Profit-taking (engine/profit_manager.py), in R = entry - initial stop ---
    # At +SCALE_OUT_AT_R sell SCALE_OUT_FRACTION and move the stop to breakeven;
    # then trail the stop TRAIL_DISTANCE_R behind the high. The target still
    # closes the rest. Stops are also held at the broker (an OCO stop + target
    # for the shares held), re-placed at most every BROKER_STOP_UPDATE_SECONDS.
    PROFIT_TAKING_ENABLED: bool = True
    SCALE_OUT_AT_R: float = 1.0
    SCALE_OUT_FRACTION: float = 0.5
    BREAKEVEN_BUFFER_PCT: float = 0.05
    TRAIL_DISTANCE_R: float = 1.0
    STOP_MIN_STEP_R: float = 0.1
    BROKER_STOP_UPDATE_SECONDS: float = 30.0
    BROKER_PROTECTION_CHECK_SECONDS: float = 60.0

    # --- Day trading: nothing is held through the close ---
    # Stocks are flattened this long before their market closes (the broker clock's
    # own close time, so half-days are respected) and get no new entries inside
    # NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE.
    DAY_TRADE_FLATTEN_ENABLED: bool = True
    FLATTEN_MINUTES_BEFORE_CLOSE: float = 10.0
    NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE: float = 30.0

    # --- Portfolio manager (engine/portfolio_manager.py) ---
    # Owns every entry: ranks the buy signals by conviction and diversification
    # fit, deploys idle budget into the best ones, and reports why any stays idle.
    MANAGER_ENABLED: bool = True
    MANAGER_INTERVAL_SECONDS: float = 2.0
    MANAGER_MAX_ENTRIES_PER_CYCLE: int = 2
    # A symbol whose entry was tried and did not end in an order (a veto, a broker
    # refusal) is left alone this long, so a standing veto is not re-sent every cycle.
    MANAGER_RETRY_SECONDS: float = 30.0
    MANAGER_MAX_TICK_AGE_SECONDS: float = 180.0
    # Weight of diversification fit against conviction when ranking entries.
    MANAGER_FIT_WEIGHT: float = 0.5

    # --- Trend analyst (engine/trend.py, core/minute_bars.py) ---
    # Reads each symbol's trend for the dashboard. It no longer gates entries:
    # the strategy that decides an entry owns all of its conditions, so what the
    # backtester and the RL environment replay is exactly what trades.
    MINUTE_BARS_BACKFILL: bool = True
    TREND_INTERVAL_SECONDS: float = 1.0
    TREND_MIN_BARS: int = 20
    TREND_SHORT_BARS: int = 15
    TREND_LONG_BARS: int = 60
    TREND_UP: float = 0.30              # |direction| at which the label becomes up/down

    # --- RL policy (rl/, engine/strategies/rl_ppo.py) ---
    # auto    trades (paper) only when the deployed policy passed its promotion
    #         gate on unseen days; otherwise runs in shadow mode
    # shadow  decides and logs every decision, never places an order (default:
    #         news sentiment trades while the policy keeps learning)
    # live    trades whatever policy is deployed, gate or not (use with care)
    # off     not consulted
    RL_MODE: str = Field(default=os.getenv("RL_MODE", "shadow"))
    # Streamed and backfilled for market context (the policy reads SPY), never traded
    # unless also on the watchlist.
    CONTEXT_SYMBOLS: tuple = ("SPY",)
    RL_GATE_MIN_T: float = 2.0          # test-set t-stat a policy needs to be approved
    # Learner agent: after each close, retrain on the newest data (warm start)
    # and deploy the result only if it beats the current policy on the same days.
    RL_RETRAIN_ENABLED: bool = True
    RL_RETRAIN_AFTER_CLOSE_MINUTES: float = 30.0
    RL_RETRAIN_ITERATIONS: int = 120

    # --- Smart money (engine/smart_money.py, engine/strategies/smart_money.py) ---
    # Tradable BUYs (sizeable insider purchases, broad top-investor ownership;
    # liquid US stocks only) are put on the watchlist and day-traded by the
    # smart_money strategy. Everything else stays with the default strategy.
    SMART_MONEY_TRADING: bool = True
    SMART_MONEY_LOOKBACK_DAYS: float = 5.0         # an insider purchase counts this long
    SMART_MONEY_MIN_BUY_USD: float = 100_000.0     # smaller purchases are noise
    SMART_MONEY_MIN_BREADTH: float = 0.20          # share of the top eToro investors holding it
    SMART_MONEY_MIN_PRICE: float = 5.0
    SMART_MONEY_MIN_DOLLAR_VOLUME: float = 10_000_000.0   # average daily $ traded
    SMART_MONEY_FIRST_ENTRY_MINUTES: float = 30.0  # no entries in the first half hour

    # --- Scout: our own stock discovery (scout/) ---
    # Every SCOUT_INTERVAL_SECONDS it ranks the stocks worth watching today from
    # past-window performance, today's move, news and public discussion (Reddit,
    # StockTwits), and puts the top SCOUT_TOP_N on the watchlist.
    SCOUT_ENABLED: bool = True
    SCOUT_INTERVAL_SECONDS: float = 3600.0
    SCOUT_TOP_N: int = 10
    # A pick stays on the watchlist until it falls out of the top
    # SCOUT_KEEP_RANK, so ranks shuffling by a place cause no churn.
    SCOUT_KEEP_RANK: int = 20
    SCOUT_MIN_SCORE: float = 0.55
    # Eligibility: a real, liquid US common stock with enough history to judge.
    SCOUT_MIN_PRICE: float = 5.0
    SCOUT_MIN_DOLLAR_VOLUME: float = 20_000_000.0
    SCOUT_MAX_POOL: int = 500
    # The world's exchanges (scout/exchanges.py): each hour, the SCOUT_EXCHANGE_TOP
    # most-traded stocks of every market listed here ("tradingview market:label"),
    # mapped by company name to a US line Alpaca can trade (an ADR or a US
    # listing). Hot stocks with no such line are shown but cannot be traded.
    SCOUT_EXCHANGES: str = ("uk:London,germany:Xetra,france:Paris,netherlands:Amsterdam,switzerland:Zurich,"
                            "hongkong:Hong Kong,india:India NSE,japan:Tokyo,korea:Korea,taiwan:Taiwan")
    SCOUT_EXCHANGE_TOP: int = 40
    # Picks reserved per exchange on top of SCOUT_TOP_N (its best tradable stock),
    # at most SCOUT_INTL_MAX_PICKS in all: foreign stocks get little US news and
    # Reddit attention and would otherwise rarely make the overall top.
    SCOUT_EXCHANGE_SLOTS: int = 1
    SCOUT_INTL_MAX_PICKS: int = 6
    SCOUT_EXCHANGE_MIN_SCORE: float = 0.45
    SCOUT_NEWS_LOOKBACK_HOURS: float = 24.0
    SCOUT_NEWS_MAX_ITEMS: int = 1000
    # Headline tone is scored (Jev/Laya) for this many of the best candidates,
    # at most SCOUT_TONE_HEADLINES of each one's newest headlines.
    SCOUT_TONE_TOP_N: int = 30
    SCOUT_TONE_HEADLINES: int = 3
    SCOUT_REDDIT_PAGES: int = 2

    # --- Watcher: keeps eyes on the scout's picks, decides when to trade ---
    # Confidence (0..1) from the pick's scout score, the intraday trend, VWAP,
    # the move since the open, fresh news and the market. An entry needs it at
    # or above SCOUT_ENTRY_CONFIDENCE without a break for SCOUT_CONFIRM_SECONDS.
    SCOUT_TRADING: bool = True
    SCOUT_WATCH_INTERVAL_SECONDS: float = 5.0
    SCOUT_ENTRY_CONFIDENCE: float = 0.65
    SCOUT_CONFIRM_SECONDS: float = 90.0
    SCOUT_EXIT_CONFIDENCE: float = 0.40
    SCOUT_MIN_HOLD_MINUTES: float = 15.0
    SCOUT_FIRST_ENTRY_MINUTES: float = 15.0      # no entries in the opening auction noise
    SCOUT_MAX_ENTRIES_PER_DAY: int = 1           # per symbol

    # --- Trade desk: every entry is observed, then argued by LLM agents (desk/) ---
    # Observer   watches the stock for DESK_OBSERVE_SECONDS after the signal; the
    #            signal must stay present for DESK_MIN_PERSISTENCE of the checks
    # Analyst    DESK_ANALYST_MODEL with reasoning: thesis and P(target before stop)
    # Critic     DESK_CRITIC_MODEL: argues against the trade, may veto
    # Decision   both agree and their mean probability clears the breakeven of the
    #            trade's own stop/target by a margin set by the risk dial (and DESK_MIN_PROB)
    # The executor refuses any buy without a fresh approval (DESK_REQUIRED), so an
    # unreachable Ollama blocks entries rather than letting them through unreviewed.
    DESK_ENABLED: bool = True
    DESK_REQUIRED: bool = True
    DESK_REVIEW_WHEN_PAUSED: bool = False
    OLLAMA_URL: str = Field(default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    # Two different models, so the critic is a second opinion and not the same
    # model agreeing with itself. On the 8 GB GPU they cannot both sit in VRAM:
    # swapping them costs 1.5-3 minutes a review, so the analyst stays wholly on
    # the GPU and the critic takes the VRAM left over, the rest of its layers in
    # RAM (DESK_CRITIC_NUM_GPU layers on the GPU; -1 lets Ollama decide).
    DESK_ANALYST_MODEL: str = "qwen3.5:9b"
    DESK_ANALYST_THINK: bool = True
    DESK_ANALYST_NUM_GPU: int = -1
    DESK_CRITIC_MODEL: str = "qwen3:4b"
    DESK_CRITIC_THINK: bool = False
    DESK_CRITIC_NUM_GPU: int = 12
    DESK_LLM_TIMEOUT_SECONDS: float = 150.0
    DESK_LLM_MAX_TOKENS: int = 3500
    # Reasoning budget (~35 tokens/s on an RTX 3070): past it the model is handed
    # its notes and asked to answer, so a review cannot run for minutes.
    DESK_THINK_BUDGET_TOKENS: int = 1000
    DESK_OBSERVE_SECONDS: float = 90.0
    DESK_MIN_PERSISTENCE: float = 0.7
    DESK_SIGNAL_GAP_SECONDS: float = 20.0      # signal missing this long while observed: faded
    # The bar is the trade's own breakeven (from its stop and target distances)
    # plus the risk dial's margin (below), never below DESK_MIN_PROB.
    DESK_MIN_PROB: float = 0.40
    # Margin over breakeven, by the risk dial: DESK_EDGE_MARGIN_CAUTIOUS at dial 1
    # down to DESK_EDGE_MARGIN_AGGRESSIVE at dial 10 (linear). A cautious user
    # needs a clearer edge before the desk approves.
    DESK_EDGE_MARGIN_CAUTIOUS: float = 0.08
    DESK_EDGE_MARGIN_AGGRESSIVE: float = 0.02
    # Context window per call: the case file, the reasoning and the answer must
    # fit, or Ollama silently drops the start of the prompt (its default is 4096).
    DESK_NUM_CTX: int = 8192
    DESK_CLEARANCE_SECONDS: float = 180.0      # an approval is good this long
    DESK_MAX_PRICE_DRIFT_PCT: float = 0.5      # ...and while price stays this close
    DESK_REJECT_COOLDOWN_SECONDS: float = 900.0
    DESK_MAX_ACTIVE_CASES: int = 4

    # --- Curator: moves discovery's best picks onto the watchlist ---
    CURATOR_INTERVAL_SECONDS: float = 30.0

    # Times shown to the operator (dashboard, logs, scheduler labels). Market
    # logic never uses it: US sessions are always computed in New York time.
    DISPLAY_TIMEZONE: str = Field(default=os.getenv("DISPLAY_TIMEZONE", "Europe/Paris"))

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    TELEMETRY_BROADCAST_MS: int = 250  # WebSocket push rate to frontend (4x per sec)

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
