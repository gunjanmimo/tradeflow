"""
Sentiment backend router: Jev first, Laya as the fallback.

Jev (TypeSafe, hosted) and Laya (in-process weights) live in separate modules
and know nothing about each other; this is the only place that chooses. Once
Jev hits a quota/auth error, or fails TYPESAFE_MAX_CONSECUTIVE_FAILURES times
in a row, the router switches to Laya for the rest of the process lifetime.
Restart the backend to try Jev again. A headline whose Jev call failed is
re-scored by Laya, so no headline is dropped during the switch.
"""
import logging
from typing import Optional

from core.config import settings
from core.state import state, SentimentRecord
from sentiment.jev_service import jev_service, JevUnavailable
from sentiment.laya_service import laya_service

logger = logging.getLogger("tradeflow.sentiment")


class SentimentRouter:
    def __init__(self):
        self.active = "laya"
        self.fallback_reason: Optional[str] = None
        self._consecutive_failures = 0

    async def initialize(self):
        # Laya is always loaded so the fallback is warm the moment Jev fails.
        await laya_service.initialize()
        if settings.SENTIMENT_BACKEND.lower() == "jev" and jev_service.configured:
            await jev_service.initialize()
            self.active = "jev"
        else:
            self.active = "laya"
            if settings.SENTIMENT_BACKEND.lower() == "jev":
                self.fallback_reason = "TYPESAFE_API_KEY not set"
        logger.info(f"Sentiment backend: {self.active}")

    def _fall_back(self, reason: str):
        self.active = "laya"
        self.fallback_reason = reason
        logger.error(f"Jev disabled, falling back to Laya: {reason}")
        state.log_event("SENTIMENT_FALLBACK", f"Jev disabled ({reason}); Laya is now the sentiment backend")

    async def score_headline(self, symbol: str, headline: str,
                             persist: bool = True) -> SentimentRecord:
        if self.active == "jev":
            try:
                rec = await jev_service.score_headline(symbol, headline, persist=persist)
                self._consecutive_failures = 0
                return rec
            except JevUnavailable as e:
                self._consecutive_failures += 1
                logger.warning(f"Jev failed for {symbol} ({self._consecutive_failures}x): {e}")
                if e.fatal:
                    self._fall_back(str(e))
                elif self._consecutive_failures >= settings.TYPESAFE_MAX_CONSECUTIVE_FAILURES:
                    self._fall_back(f"{self._consecutive_failures} consecutive failures, last: {e}")
        return await laya_service.score_headline(symbol, headline, persist=persist)

    def status(self) -> dict:
        return {"active": self.active, "fallback_reason": self.fallback_reason}

    async def close(self):
        await jev_service.close()


sentiment_service = SentimentRouter()
