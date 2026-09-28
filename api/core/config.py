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
    # Manager: refuse a new entry when regime-suited strategies clearly oppose it.
    COUNCIL_ENTRY_CHECK: bool = True
    # Trade bots: exit a held position when the council turns decisively bearish.
    COUNCIL_EXIT_CHECK: bool = True
    COUNCIL_MIN_VOTERS: int = 3
    COUNCIL_SUPPORT_CONSENSUS: float = 0.20
    COUNCIL_VETO_CONSENSUS: float = -0.25   # entry refused at or below this
    COUNCIL_EXIT_CONSENSUS: float = -0.50   # held position closed at or below this

    # --- Off-process analysis worker (engine/analysis) ---
    # Everything heavier than a dict lookup runs in a separate process on this
    # cadence, so it can never add latency to the tick path.
    ANALYSIS_ENABLED: bool = True
    ANALYSIS_INTERVAL_SECONDS: float = 1.0
    # Results older than this are ignored by the hot path (treated as absent).
    ANALYSIS_MAX_AGE_SECONDS: float = 10.0
    # Adaptive strategy: how many worker-ranked candidates to evaluate per tick.
    ADAPTIVE_TOP_K: int = 3
    # Monte Carlo: minimum P(take-profit before stop) to allow an entry.
    # 0 disables the gate (the probability is still computed and shown).
    MC_MIN_TP_FIRST_PROB: float = 0.0
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
    DISCOVERY_AUTO_PROMOTE: bool = True
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
    PREMARKET_TRADING_ENABLED: bool = True
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
    # A forced exit never sits waiting longer than this: a failed close is retried
    # within it, whatever the normal backoff would have been.
    FORCED_EXIT_MAX_WAIT_SECONDS: float = 50.0

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
    # Nothing is bought until a symbol has TREND_MIN_BARS one-minute bars behind
    # it (backfilled from Alpaca, so normally at once) and its trend reads up.
    MINUTE_BARS_BACKFILL: bool = True
    TREND_INTERVAL_SECONDS: float = 1.0
    TREND_MIN_BARS: int = 20
    TREND_SHORT_BARS: int = 15
    TREND_LONG_BARS: int = 60
    TREND_UP: float = 0.30              # |direction| at which the label becomes up/down
    TREND_ENTRY_MIN: float = 0.25       # new entries need at least this direction

    # --- Position manager: trend-driven BUY more / SELL part / HOLD / CLOSE ---
    # Tick-level stop, target, stale-price and end-of-day exits stay with each
    # position's sentinel; this agent acts on the trend, every few seconds.
    POSITION_MANAGER_ENABLED: bool = True
    POSITION_MANAGER_INTERVAL_SECONDS: float = 2.0
    TREND_EXIT_DIRECTION: float = 0.45  # close when direction <= -this (confident)
    TREND_TRIM_DIRECTION: float = 0.25  # trim a winner when direction <= -this, or on a reversal
    TRIM_FRACTION: float = 0.5

    # Profit harvest: whenever an open position shows ANY profit at the bid (the
    # price a sell fills at), sell PROFIT_HARVEST_FRACTION of it and book the gain
    # as day income. Income is ring-fenced: never traded again, never cushions a
    # loss. The remainder is harvested again only on new profit (a bid above the
    # last harvest). PROFIT_HARVEST_USD > 0 sets a minimum; 0 = any profit.
    # Never fires on a loss. A position too small to split is held whole.
    PROFIT_HARVEST_ENABLED: bool = True
    PROFIT_HARVEST_USD: float = 0.0
    PROFIT_HARVEST_FRACTION: float = 0.5
    # A harvest must bank at least this much (net of fees): no $0.00 sells.
    PROFIT_HARVEST_MIN_INCOME: float = 0.01
    PROFIT_HARVEST_RETRY_SECONDS: float = 10.0  # e.g. a stock outside regular hours
    # Stocks: harvest in fractional shares where Alpaca allows it (a 1-share
    # winner sells 0.5), down to this many decimals and at least this notional.
    HARVEST_FRACTION_DECIMALS: int = 4
    HARVEST_MIN_FRACTIONAL_NOTIONAL: float = 1.0
    SCALE_IN_ENABLED: bool = True
    SCALE_IN_DIRECTION: float = 0.50    # add to a winner only in a strong, confident uptrend
    SCALE_IN_MIN_R: float = 1.0         # ...already up at least 1x its initial risk
    SCALE_IN_FRACTION: float = 0.5      # add this share of the original investment, once

    # --- Loss recovery (engine/loss_recovery.py) ---
    # A losing position is not dumped on one print through its stop or on a
    # headline alone. The stop has to hold for STOP_CONFIRM_SECONDS, or the price
    # has to fall STOP_DISASTER_EXTRA_R x the stop distance further (the broker's
    # bracket leg sits there). Bearish news closes a loser only when the trend
    # agrees. A stalled loser gets one rescue add that never moves the stop and
    # caps the loss at RECOVERY_MAX_RISK_MULT x the original risk, and once it is
    # back above break-even its stop is locked there.
    STOP_CONFIRM_ENABLED: bool = True
    STOP_CONFIRM_SECONDS: float = 20.0
    STOP_DISASTER_EXTRA_R: float = 0.5
    RECOVERY_ENABLED: bool = True
    RECOVERY_ADD_ENABLED: bool = True
    RECOVERY_ADD_MIN_R: float = 0.4       # rescue only once down at least this x the stop distance
    RECOVERY_ADD_MAX_R: float = 0.8       # ...and not this close to the stop
    RECOVERY_ADD_FRACTION: float = 0.5    # add at most this share of the current quantity
    RECOVERY_MAX_RISK_MULT: float = 1.25  # loss at the stop after the add <= this x original risk
    RECOVERY_MIN_MICRO: float = 0.0       # micro trend must be at least flat: the fall has stalled
    RECOVERY_RETRY_SECONDS: float = 30.0
    RECOVERY_BREAKEVEN_BUFFER_PCT: float = 0.10  # lock the stop this far above break-even (fees, spread)

    # --- Trade scorer (ml/): a small LSTM that learns from every closed trade ---
    # off     not consulted
    # shadow  scores every entry candidate and records it; decides nothing (default)
    # rank    also blends the score into the manager's ranking (ML_RANK_WEIGHT)
    # gate    also skips candidates scored under the model's own threshold
    # Experience (entry bars + final net result) is recorded in every mode but off.
    ML_MODE: str = Field(default=os.getenv("ML_MODE", "shadow"))
    ML_RECORD_EXPERIENCE: bool = True
    ML_RANK_WEIGHT: float = 0.3

    # --- Curator: moves discovery's best picks onto the watchlist ---
    CURATOR_INTERVAL_SECONDS: float = 30.0

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    TELEMETRY_BROADCAST_MS: int = 250  # WebSocket push rate to frontend (4x per sec)

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
