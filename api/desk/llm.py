"""
A small streaming client for Ollama's /api/chat.

Reasoning ("thinking") and the answer arrive as separate streams; both are
handed to callbacks as they come, so the dashboard can show a model thinking
live. The answer is constrained to a JSON schema and parsed.
"""
import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from core.config import settings

logger = logging.getLogger("tradeflow.desk.llm")


class LLMError(Exception):
    pass


class OllamaClient:
    def __init__(self):
        self._health: Dict[str, Any] = {"ok": False, "detail": "not checked yet", "models": [], "at": 0.0}

    @property
    def url(self) -> str:
        return settings.OLLAMA_URL.rstrip("/")

    async def health(self, max_age_s: float = 30.0) -> Dict[str, Any]:
        """Reachable, and are the configured models pulled? Cached for max_age_s."""
        if time.time() - self._health["at"] < max_age_s:
            return self._health
        need = sorted({settings.DESK_ANALYST_MODEL, settings.DESK_CRITIC_MODEL})
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
                async with s.get(f"{self.url}/api/tags") as r:
                    data = await r.json()
            models = [m["name"] for m in data.get("models", [])]
            missing = [m for m in need if m not in models]
            self._health = {"ok": not missing, "models": models, "at": time.time(),
                            "detail": f"missing model(s): {', '.join(missing)} (ollama pull ...)" if missing
                            else f"{self.url} ({', '.join(need)})"}
        except Exception as e:
            self._health = {"ok": False, "models": [], "at": time.time(),
                            "detail": f"Ollama unreachable at {self.url}: {type(e).__name__}"}
        return self._health

    async def chat(self, model: str, messages: List[Dict[str, str]], schema: Dict[str, Any], think: bool,
                   on_thinking: Optional[Callable[[str], None]] = None,
                   on_content: Optional[Callable[[str], None]] = None,
                   max_tokens: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Streams one chat completion. Returns {"answer": parsed JSON or None, "thinking":
        str, "tokens": int, "seconds": float, "truncated": bool}. "truncated" means the
        token budget ran out while the model was still reasoning: answer is None and
        the caller may ask for the answer from the notes (see answer_from_notes).
        Raises LLMError on any other failure, or on an answer that is not JSON.
        """
        body = {"model": model, "messages": messages, "stream": True, "think": think, "format": schema,
                "keep_alive": "15m",
                "options": {"temperature": 0.2, "num_predict": max_tokens or settings.DESK_LLM_MAX_TOKENS,
                            "num_ctx": settings.DESK_NUM_CTX, **(options or {})}}
        truncated = False
        thinking, content, tokens, t0 = [], [], 0, time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=settings.DESK_LLM_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(f"{self.url}/api/chat", json=body) as r:
                    if r.status != 200:
                        raise LLMError(f"HTTP {r.status}: {(await r.text())[:200]}")
                    async for raw in r.content:
                        line = raw.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        if d.get("error"):
                            raise LLMError(d["error"])
                        m = d.get("message") or {}
                        if m.get("thinking"):
                            thinking.append(m["thinking"])
                            if on_thinking:
                                on_thinking(m["thinking"])
                        if m.get("content"):
                            content.append(m["content"])
                            if on_content:
                                on_content(m["content"])
                        if d.get("done"):
                            tokens = int(d.get("eval_count") or 0)
                            truncated = d.get("done_reason") == "length"
        except asyncio.TimeoutError:
            raise LLMError(f"timed out after {settings.DESK_LLM_TIMEOUT_SECONDS:.0f}s")
        except aiohttp.ClientError as e:
            raise LLMError(f"{type(e).__name__}: {e}")
        text = "".join(content).strip()
        out = {"answer": None, "thinking": "".join(thinking), "tokens": tokens,
               "seconds": round(time.time() - t0, 1), "truncated": False}
        if truncated and not text:
            out["truncated"] = True
            return out
        try:
            out["answer"] = json.loads(text)
        except ValueError:
            raise LLMError(f"answer is not JSON: {text[:160]!r}")
        return out

    async def answer_from_notes(self, model: str, messages: List[Dict[str, str]], schema: Dict[str, Any],
                                notes: str, on_content: Optional[Callable[[str], None]] = None,
                                options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The reasoning budget ran out: hand the model its own notes and ask for the answer, no more thinking."""
        follow = messages + [{"role": "user", "content":
                              "Your reasoning budget is used up. Your notes so far:\n"
                              f"{notes[-6000:]}\n\nStop reasoning and give your final answer now in the JSON schema."}]
        out = await self.chat(model, follow, schema, think=False, on_content=on_content, max_tokens=1500,
                              options=options)
        if out["answer"] is None:
            raise LLMError("no answer even without reasoning")
        return out


ollama = OllamaClient()
