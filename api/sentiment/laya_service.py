import asyncio
import time
import logging
from typing import Optional, Dict, Any
from core.state import state, SentimentRecord

logger = logging.getLogger("tradeflow.laya")

class LayaSentimentService:
    """
    Sub-second in-process sentiment inference using Convai Innovations' Laya model.
    Runs asynchronously in the background so it never blocks the price tick execution loop.
    """
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self._load_lock = asyncio.Lock()

    async def initialize(self):
        """Lazy load Laya model on server startup"""
        if self.is_loaded:
            return
        
        async with self._load_lock:
            if self.is_loaded:
                return
            try:
                logger.info("Initializing in-process Laya decision model (convaiinnovations/laya)...")
                # Run the model import and load in a separate thread to not block event loop
                loop = asyncio.get_running_loop()
                self.model = await loop.run_in_executor(None, self._load_laya_sync)
                self.is_loaded = True
                logger.info("Laya model successfully loaded into memory.")
                state.log_event("INFO", "Laya model loaded into in-process memory")
            except Exception as e:
                logger.warning(f"Could not load local weights immediately ({e}). Using fast calibrated heuristic fallback.")
                self.is_loaded = False

    def _load_laya_sync(self):
        import laya
        # Pass official open Hugging Face repo ID so snapshot_download finds public weights
        repo_id = "convaiinnovations/laya"
        return laya.load(repo_id)

    def score_text_sync(self, symbol: str, headline: str, persist: bool = True) -> SentimentRecord:
        """
        Synchronous scoring function called in worker pool.
        Executes in ~15-35ms.
        """
        t0 = time.time()
        pos_prob = 0.5
        neg_prob = 0.5
        open_flag = False
        degraded = False

        if self.is_loaded and self.model is not None:
            try:
                questions = {
                    "sentiment": {
                        "type": "choice",
                        "instructions": f"What is the financial market sentiment for {symbol}?",
                        "criteria": ["bullish", "bearish", "neutral"]
                    }
                }
                res = self.model.predict(headline, questions)
                answers = res.get("answers", {}) if isinstance(res, dict) else getattr(res, "answers", {})
                
                sentiment_ans = answers.get("sentiment", {})
                probs = sentiment_ans.get("probabilities", {}) if isinstance(sentiment_ans, dict) else getattr(sentiment_ans, "probabilities", {})
                pos_prob = float(probs.get("bullish", 0.5))
                neg_prob = float(probs.get("bearish", 0.5))
                open_flag = pos_prob > 0.60 or neg_prob > 0.60
            except Exception as e:
                logger.error(f"Error evaluating Laya on text: {e}")
                # Fail NEUTRAL, not to a keyword guess. A keyword tally dressed up
                # as model output is worse than no signal: 0.5/0.5 fails the entry
                # gates, whereas a guessed 0.80 would authorise a real trade.
                pos_prob, neg_prob, degraded = 0.5, 0.5, True
        else:
            pos_prob, neg_prob, degraded = 0.5, 0.5, True

        open_flag = pos_prob >= 0.60 or neg_prob >= 0.60
        latency_ms = (time.time() - t0) * 1000

        record = SentimentRecord(
            stock_id=symbol,
            pos_prob=round(pos_prob, 4),
            neg_prob=round(neg_prob, 4),
            open=open_flag,
            headline=headline[:120],
            updated_at=time.time()
        )
        
        # Callers that aggregate headlines pass persist=False and record the
        # score themselves; only the legacy single-write path mutates the cache.
        if persist:
            state.update_sentiment(record)
        if degraded:
            state.log_event(
                "SENTIMENT_DEGRADED",
                f"Laya unavailable for {symbol}; returned NEUTRAL 0.5/0.5 (entries blocked)."
            )
        else:
            state.log_event(
                "SENTIMENT",
                f"Laya scored {symbol} in {latency_ms:.1f}ms: pos={pos_prob:.2f}, neg={neg_prob:.2f}"
            )
        return record

    async def score_headline(self, symbol: str, headline: str,
                             persist: bool = True) -> SentimentRecord:
        """Async non-blocking scoring entry point"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.score_text_sync, symbol, headline, persist
        )

laya_service = LayaSentimentService()
