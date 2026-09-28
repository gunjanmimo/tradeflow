import time
import logging
from typing import Optional

import aiohttp

from core.config import settings
from core.state import state, SentimentRecord

logger = logging.getLogger("tradeflow.jev")

# HTTP statuses that mean Jev will not recover by retrying: bad/revoked key,
# no access, or the account's usage limit is exhausted.
FATAL_STATUSES = {401, 402, 403, 429}


class JevUnavailable(Exception):
    """Jev could not score a headline. `fatal` means stop calling Jev."""

    def __init__(self, message: str, fatal: bool = False, status: Optional[int] = None):
        super().__init__(message)
        self.fatal = fatal
        self.status = status


class JevSentimentService:
    """
    Sentiment inference via TypeSafe's hosted Jev model (System One API).
    Same question shape as Laya: one Choice over bullish/bearish/neutral.
    Unlike Laya it never fails neutral on its own: every failure raises
    JevUnavailable so the router can hand the headline to Laya instead.
    """
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def configured(self) -> bool:
        return bool(settings.TYPESAFE_API_KEY)

    async def initialize(self):
        if not self.configured:
            logger.warning("TYPESAFE_API_KEY not set; Jev disabled.")
            return
        logger.info(f"Jev sentiment backend ready ({settings.TYPESAFE_MODEL}).")
        state.log_event("INFO", f"Jev ({settings.TYPESAFE_MODEL}) is the sentiment backend")

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=settings.TYPESAFE_TIMEOUT_SECONDS),
                headers={
                    "Authorization": f"Bearer {settings.TYPESAFE_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def score_headline(self, symbol: str, headline: str,
                             persist: bool = True) -> SentimentRecord:
        if not self.configured:
            raise JevUnavailable("TYPESAFE_API_KEY not set", fatal=True)

        t0 = time.time()
        payload = {
            "state": headline,
            "model": settings.TYPESAFE_MODEL,
            "questions": {
                "sentiment": {
                    "type": "choice",
                    "instructions": f"What is the financial market sentiment for {symbol}?",
                    "criteria": {"bullish": None, "bearish": None, "neutral": None},
                }
            },
        }
        try:
            async with self._get_session().post(settings.TYPESAFE_BASE_URL, json=payload) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    raise JevUnavailable(
                        f"HTTP {resp.status}: {body}",
                        fatal=resp.status in FATAL_STATUSES,
                        status=resp.status,
                    )
                data = await resp.json()
        except JevUnavailable:
            raise
        except Exception as e:
            raise JevUnavailable(f"{type(e).__name__}: {e}") from e

        try:
            probs = data["answers"]["sentiment"]["probabilities"]
            pos_prob = float(probs.get("bullish", 0.0))
            neg_prob = float(probs.get("bearish", 0.0))
        except (KeyError, TypeError, ValueError) as e:
            raise JevUnavailable(f"Malformed Jev response: {e}") from e

        latency_ms = (time.time() - t0) * 1000
        record = SentimentRecord(
            stock_id=symbol,
            pos_prob=round(pos_prob, 4),
            neg_prob=round(neg_prob, 4),
            open=pos_prob >= 0.60 or neg_prob >= 0.60,
            headline=headline[:120],
            updated_at=time.time(),
        )
        if persist:
            state.update_sentiment(record)
        state.log_event(
            "SENTIMENT",
            f"Jev scored {symbol} in {latency_ms:.1f}ms: pos={pos_prob:.2f}, neg={neg_prob:.2f}"
        )
        return record


jev_service = JevSentimentService()
