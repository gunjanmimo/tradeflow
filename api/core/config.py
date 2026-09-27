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

    # Stop-distance bounds, as a fraction of price. Both are relative, never absolute
    # dollar amounts -- an absolute floor is meaningless across assets priced from
    # $0.00002 (SHIB) to $85,000 (BTC).
    MIN_STOP_DISTANCE_PCT: float = 0.008  # never risk a stop tighter than 0.8%
    MAX_STOP_DISTANCE_PCT: float = 0.06   # never accept a stop wider than 6%
    MAX_POSITION_NOTIONAL_PCT: float = 0.15  # max 15% of budget in one position

    # Portfolio-level circuit breakers (evaluated before every new entry)
    MAX_DAILY_LOSS_PERCENT: float = 3.0   # halt new entries after -3% on the day
    MAX_DRAWDOWN_PERCENT: float = 10.0    # halt new entries after -10% from peak equity
    MAX_POSITIONS_PER_ASSET_CLASS: int = 3  # cap correlated exposure (e.g. 5 L1 tokens)

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

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    TELEMETRY_BROADCAST_MS: int = 250  # WebSocket push rate to frontend (4x per sec)

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
