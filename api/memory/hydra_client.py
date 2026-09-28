"""
HydraDB client: async OpenCypher over the HTTP query API.

Engine constraints this wraps (verified against hydradb 0.1.0, not guesses)
-------------------------------------------------------------------------
1. Every node needs an integer `id` property; it becomes the vertex id. String
   keys are hashed to a stable int64 by `stable_id()`.
2. `CREATE` only accepts ONE-HOP EDGE PATTERNS: `(a {id:N})-[:R]->(b {id:M})`.
   A bare single-node CREATE is rejected ("only one-hop edge patterns are
   executable"), so standalone nodes are attached to an anchor node instead.
3. `MERGE` is unsupported. Upsert is emulated by inlining all properties in
   CREATE; re-creating an existing id is idempotent at the vertex level.
4. Aggregates are unreliable -- `count()` errors outright and `avg()` demands
   integers. All statistics are therefore computed in Python over returned rows.
   Trade volumes are in the thousands, so this is not a performance concern.
5. A node-only `MATCH` needs an id, label, or property predicate.
6. `MATCH…SET` triggers `PutMode::Update` in the underlying SlateDB storage
   layer, which the LocalFileSystem object-store backend does NOT implement.
   All property mutations must therefore be done by inlining props in CREATE
   (for new nodes) or DELETE-then-CREATE (for existing nodes).

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
        return (await self._execute(cypher, params, quiet))[1]

    async def write(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Runs a write statement; True only if the engine accepted it. A CREATE
        returns no rows either way, so query()'s [] cannot tell success from failure.
        """
        return (await self._execute(cypher, params, quiet=True))[0]

    async def _execute(self, cypher: str, params: Optional[Dict[str, Any]],
                       quiet: bool) -> "tuple[bool, List[Dict[str, Any]]]":
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
                    return False, []

                self.queries_ok += 1
                self.is_available = True
                cols = body.get("columns", []) or []
                rows = body.get("rows", []) or []
                return True, [
                    {cols[i]: self._unwrap(cell) for i, cell in enumerate(row) if i < len(cols)}
                    for row in rows
                ]
        except Exception as e:
            self.last_error = str(e)
            self.queries_failed += 1
            self.is_available = False
            if not quiet:
                logger.debug(f"HydraDB unreachable: {e}")
            return False, []

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_props(props: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and coerce property values to engine-accepted scalars."""
        clean = {}
        for k, v in props.items():
            if v is None:
                continue
            if isinstance(v, bool) or isinstance(v, (int, float, str)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    async def upsert_node(self, label: str, key: str,
                          props: Dict[str, Any]) -> int:
        """
        Creates or updates a node identified by `key`.

        Properties are embedded directly in the CREATE statement to avoid the
        MATCH…SET pattern, which requires PutMode::Update — an operation the
        LocalFileSystem object-store backend does not implement. Re-creating an
        existing vertex id is idempotent at the vertex level, and inline
        properties are applied on creation.
        """
        node_id = stable_id(key)
        anchor_id = stable_id(self.ANCHOR_KEY)

        all_props = self._clean_props({**props, "key": key, "id": node_id})

        # Build inline property map: {id: $nid, key: $key, bot_id: $bot_id, …}
        prop_fragment = ", ".join(f"{k}: ${k}" for k in all_props)
        await self.query(
            f"CREATE (a:Anchor {{id: $anchor}})-[:HAS]->"
            f"(n:{label} {{{prop_fragment}}})",
            {"anchor": anchor_id, **all_props}, quiet=True,
        )
        return node_id

    async def set_props(self, label: str, node_id: int, props: Dict[str, Any]) -> bool:
        """
        Update a node's properties.

        The LocalFileSystem object-store backend does not support
        PutMode::Update, which makes MATCH…SET fail. As a workaround we
        DELETE the node and re-CREATE it via the anchor with all properties
        inlined in the CREATE statement.
        """
        clean = self._clean_props(props)
        if not clean:
            return False

        # First, read current properties so we can merge old + new.
        rows = await self.query(
            f"MATCH (n:{label} {{id: $nid}}) RETURN n",
            {"nid": node_id}, quiet=True,
        )
        existing: Dict[str, Any] = {}
        if rows:
            raw = rows[0].get("n", {})
            if isinstance(raw, dict):
                existing = {k: self._unwrap(v) for k, v in raw.items()}

        merged = {**existing, **clean, "id": node_id}

        # Delete old node (DETACH DELETE removes the node and its edges).
        await self.query(
            f"MATCH (n:{label} {{id: $nid}}) DETACH DELETE n",
            {"nid": node_id}, quiet=True,
        )

        # Re-create with anchor edge and all properties inline.
        anchor_id = stable_id(self.ANCHOR_KEY)
        prop_fragment = ", ".join(f"{k}: ${k}" for k in merged)
        await self.query(
            f"CREATE (a:Anchor {{id: $anchor}})-[:HAS]->"
            f"(n:{label} {{{prop_fragment}}})",
            {"anchor": anchor_id, **merged}, quiet=True,
        )
        return True

    async def link(self, from_label: str, from_key: str, rel: str,
                   to_label: str, to_key: str,
                   from_props: Optional[Dict[str, Any]] = None,
                   to_props: Optional[Dict[str, Any]] = None) -> bool:
        """
        Creates an edge, materialising both endpoints in the same statement --
        which is exactly the one-hop pattern the engine supports.

        All properties are inlined in the CREATE to avoid MATCH…SET (which
        requires PutMode::Update, unsupported by LocalFileSystem).
        """
        fid, tid = stable_id(from_key), stable_id(to_key)

        from_all = self._clean_props({**(from_props or {}), "key": from_key, "id": fid})
        to_all = self._clean_props({**(to_props or {}), "key": to_key, "id": tid})

        # Build inline property fragments
        from_frag = ", ".join(f"{k}: $from_{k}" for k in from_all)
        to_frag = ", ".join(f"{k}: $to_{k}" for k in to_all)

        params: Dict[str, Any] = {}
        params.update({f"from_{k}": v for k, v in from_all.items()})
        params.update({f"to_{k}": v for k, v in to_all.items()})

        return await self.write(
            f"CREATE (a:{from_label} {{{from_frag}}})"
            f"-[:{rel}]->"
            f"(b:{to_label} {{{to_frag}}})",
            params,
        )

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
        self_id = stable_id(self.ANCHOR_KEY + ":self")
        # Deterministic payload: HydraDB restarts its request ids at http-query-1
        # after a restart, so a timestamp here made the same idempotency key arrive
        # with a different payload and the anchor write was rejected as a conflict.
        await self.query(
            "CREATE (a:Anchor {id: $id, key: $key})"
            "-[:SELF]->"
            "(b:Anchor {id: $id2, key: $key2})",
            {"id": anchor_id, "key": self.ANCHOR_KEY,
             "id2": self_id, "key2": self.ANCHOR_KEY + ":self"}, quiet=True,
        )
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
