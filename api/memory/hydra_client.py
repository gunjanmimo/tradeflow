"""
HydraDB client: async OpenCypher over the HTTP query API.

Engine constraints this wraps (verified against hydradb 0.1.0, not guesses)
-------------------------------------------------------------------------
1. Every node needs an integer `id` property; it becomes the vertex id. String
   keys are hashed to a stable int64 by `stable_id()`.
2. `CREATE` only accepts ONE-HOP EDGE PATTERNS: `(a {id:N})-[:R]->(b {id:M})`.
   A bare single-node CREATE is rejected ("only one-hop edge patterns are
   executable"), so standalone nodes are attached to an anchor node instead.
3. `MERGE` is unsupported. Upsert is emulated as CREATE-then-SET; re-creating an
   existing id is idempotent at the vertex level.
4. Aggregates are unreliable -- `count()` errors outright and `avg()` demands
   integers. All statistics are therefore computed in Python over returned rows.
   Trade volumes are in the thousands, so this is not a performance concern.
5. A node-only `MATCH` needs an id, label, or property predicate.

Failure policy
--------------
Memory is strictly OPTIONAL. Every call swallows its errors and returns a neutral
result, because a graph outage must never block or slow a trading decision. The
tick path is never allowed to await a network round-trip to this service.
"""
import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("tradeflow.memory")


def stable_id(key: str) -> int:
    """
    Deterministic positive int64 from a string key.

    Must be stable across processes and restarts: it is the primary key for every
    node, so a changing hash would orphan all prior memory.
    """
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFF_FFFF_FFFF_FFFF


class HydraClient:
    # Anchor node that standalone vertices attach to, working around the
    # one-hop-edge-only CREATE restriction.
    ANCHOR_KEY = "tradeflow:anchor"

    def __init__(self):
        self.base_url = os.getenv("HYDRA_HTTP_URL", "http://hydradb:8443")
        self.graph_id = os.getenv("HYDRA_GRAPH_ID", "default")
        self.namespace = os.getenv("HYDRA_NAMESPACE", "default")
        self.cell_id = os.getenv("HYDRA_CELL_ID", "cell-0")
        self.token = self._load_token()
        self.is_available = False
        self.last_error: Optional[str] = None
        self.queries_ok = 0
        self.queries_failed = 0
        self._session: Optional[aiohttp.ClientSession] = None

    def _load_token(self) -> str:
        tok = os.getenv("HYDRA_AUTH_TOKEN", "").strip()
        if tok:
            return tok
        path = os.getenv("HYDRA_AUTH_TOKEN_FILE", "/data/auth-token")
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return ""

    @property
    def _query_url(self) -> str:
        return f"{self.base_url}/v1/graphs/{self.graph_id}/query"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Graph-Namespace": self.namespace,
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap(value: Any) -> Any:
        """
        HydraDB returns each cell as {"type": "...", "value": ...}. Flatten it so
        callers deal in plain Python values.
        """
        if isinstance(value, dict) and "value" in value and "type" in value:
            return value["value"]
        return value

    async def query(self, cypher: str, params: Optional[Dict[str, Any]] = None,
                    quiet: bool = False) -> List[Dict[str, Any]]:
        """
        Runs a Cypher statement. Returns a list of row dicts, or [] on any failure.

        Never raises: a memory outage must degrade recall, not halt trading.
        """
        payload: Dict[str, Any] = {"cell_id": self.cell_id, "query": cypher}
        if params:
            payload["parameters"] = params

        try:
            session = await self._get_session()
            async with session.post(self._query_url, headers=self._headers(),
                                    json=payload) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200 or "error" in body:
                    msg = (body.get("error", {}) or {}).get("message", f"HTTP {resp.status}")
                    self.last_error = msg
                    self.queries_failed += 1
                    if not quiet:
                        logger.warning(f"HydraDB query rejected: {msg} | {cypher[:120]}")
                    return []

                self.queries_ok += 1
                self.is_available = True
                cols = body.get("columns", []) or []
                rows = body.get("rows", []) or []
                return [
                    {cols[i]: self._unwrap(cell) for i, cell in enumerate(row) if i < len(cols)}
                    for row in rows
                ]
        except Exception as e:
            self.last_error = str(e)
            self.queries_failed += 1
            self.is_available = False
            if not quiet:
                logger.debug(f"HydraDB unreachable: {e}")
            return []

    # ------------------------------------------------------------------
    async def upsert_node(self, label: str, key: str,
                          props: Dict[str, Any]) -> int:
        """
        Creates or updates a node identified by `key`.

        Emulates MERGE (unsupported by the engine) as CREATE-via-anchor then SET.
        Re-running is safe: creating an existing vertex id does not duplicate it,
        and SET then reconciles the properties.
        """
        node_id = stable_id(key)
        anchor_id = stable_id(self.ANCHOR_KEY)

        # One-hop edge CREATE is the only executable form, so every standalone
        # node is born attached to the anchor.
        await self.query(
            f"CREATE (a:Anchor {{id: $anchor}})-[:HAS]->(n:{label} {{id: $nid}})",
            {"anchor": anchor_id, "nid": node_id}, quiet=True,
        )
        await self.set_props(label, node_id, {**props, "key": key})
        return node_id

    async def set_props(self, label: str, node_id: int, props: Dict[str, Any]) -> bool:
        """SET a node's properties. Only scalar types are accepted by the engine."""
        clean = {}
        for k, v in props.items():
            if v is None:
                continue
            if isinstance(v, bool) or isinstance(v, (int, float, str)):
                clean[k] = v
            else:
                clean[k] = str(v)
        if not clean:
            return False
        assignments = ", ".join(f"n.{k} = ${k}" for k in clean)
        rows = await self.query(
            f"MATCH (n:{label} {{id: $nid}}) SET {assignments}",
            {"nid": node_id, **clean},
        )
        return rows is not None

    async def link(self, from_label: str, from_key: str, rel: str,
                   to_label: str, to_key: str,
                   from_props: Optional[Dict[str, Any]] = None,
                   to_props: Optional[Dict[str, Any]] = None) -> bool:
        """
        Creates an edge, materialising both endpoints in the same statement --
        which is exactly the one-hop pattern the engine supports.
        """
        fid, tid = stable_id(from_key), stable_id(to_key)
        ok = await self.query(
            f"CREATE (a:{from_label} {{id: $fid}})-[:{rel}]->(b:{to_label} {{id: $tid}})",
            {"fid": fid, "tid": tid}, quiet=True,
        ) is not None
        if from_props:
            await self.set_props(from_label, fid, {**from_props, "key": from_key})
        if to_props:
            await self.set_props(to_label, tid, {**to_props, "key": to_key})
        return ok

    # ------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        rows = await self.query(
            "MATCH (a:Anchor {id: $id}) RETURN a.id AS id",
            {"id": stable_id(self.ANCHOR_KEY)}, quiet=True,
        )
        # An empty result is still a successful round-trip on a fresh graph, so
        # availability is judged by whether the request itself succeeded.
        return {
            "available": self.is_available,
            "url": self.base_url,
            "graph": self.graph_id,
            "queries_ok": self.queries_ok,
            "queries_failed": self.queries_failed,
            "last_error": self.last_error,
            "anchor_present": bool(rows),
        }

    async def initialize(self) -> bool:
        """Creates the anchor node and confirms reachability."""
        anchor_id = stable_id(self.ANCHOR_KEY)
        await self.query(
            "CREATE (a:Anchor {id: $id})-[:SELF]->(b:Anchor {id: $id2})",
            {"id": anchor_id, "id2": stable_id(self.ANCHOR_KEY + ":self")}, quiet=True,
        )
        await self.set_props("Anchor", anchor_id,
                             {"key": self.ANCHOR_KEY, "created_at": time.time()})
        rows = await self.query(
            "MATCH (a:Anchor {id: $id}) RETURN a.id AS id", {"id": anchor_id}, quiet=True,
        )
        self.is_available = bool(rows) or self.queries_ok > 0
        if self.is_available:
            logger.info(f"HydraDB memory layer connected at {self.base_url}")
        else:
            logger.warning(
                f"HydraDB not reachable at {self.base_url}: {self.last_error}. "
                f"Trading continues without memory."
            )
        return self.is_available


hydra = HydraClient()
