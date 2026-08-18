"""Serving-endpoint orchestrator — Databricks App (FastAPI).

A config-driven compute graph over Databricks model serving endpoints. `graph.yaml`
declares typed `nodes` and the edges between them (a node's `inputs`); a request to
`POST /graph/{name}` runs the ancestors of that graph's output node in topological
order, feeding each node's result downstream.

Node types live in NODE_TYPES and are pluggable — serving endpoints are one type;
`transform` (pure-python) and `constant` are included as examples of "other nodes".
Backend calls go to  https://<host>/serving-endpoints/<endpoint><path>  so any
contract works (MLflow `/invocations`, OpenAI-compatible `/v1/chat/completions`, or a
custom FastAPI route on an express deployment).

Auth: the app's injected service principal. `WorkspaceClient()` reads
DATABRICKS_HOST / DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET; `w.config.
authenticate()` yields a fresh bearer header per request.

Env (app.yaml):
  CONFIG_PATH   path to the graph spec (default: graph.yaml next to this file)
"""
from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import yaml
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from databricks.sdk import WorkspaceClient

CONFIG_PATH = os.environ.get("CONFIG_PATH", str(Path(__file__).with_name("graph.yaml")))
TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "300"))

_w = WorkspaceClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled HTTP client for the app's lifetime, so connections to the serving
    # host are reused across requests instead of reopened each call.
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        app.state.client = client
        yield


app = FastAPI(title="Serving Endpoint Orchestrator", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Config — parsed once, reloaded only when graph.yaml changes on disk (so edits
# still take effect without a restart, without re-parsing on every request).
# --------------------------------------------------------------------------- #
_cfg_cache: dict = {"mtime": None, "data": None}


def load_config() -> dict:
    mtime = os.path.getmtime(CONFIG_PATH)
    if _cfg_cache["mtime"] != mtime:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        _cfg_cache["data"] = {
            "nodes": {n["id"]: n for n in cfg.get("nodes", [])},
            "graphs": cfg.get("graphs", {}) or {},
        }
        _cfg_cache["mtime"] = mtime
    return _cfg_cache["data"]


# --------------------------------------------------------------------------- #
# Node-type handlers — add a new node type by registering a handler here.
# A handler is async: (node_spec, resolved_input, ctx) -> output
#   node_spec       the node's dict from graph.yaml
#   resolved_input  graph body for source nodes; {upstream_id: output} otherwise
#   ctx             {"client": httpx.AsyncClient}
# --------------------------------------------------------------------------- #
async def _node_serving_endpoint(node: dict, resolved_input: Any, ctx: dict) -> Any:
    # If fed by a single upstream node, forward that node's output as the payload.
    payload = resolved_input
    if isinstance(resolved_input, dict) and len(resolved_input) == 1 and node.get("inputs"):
        payload = next(iter(resolved_input.values()))
    host = _w.config.host.rstrip("/")
    url = f"{host}/serving-endpoints/{node['endpoint']}{node.get('path', '/invocations')}"
    headers = {"Content-Type": "application/json", **_w.config.authenticate()}
    resp = await ctx["client"].post(url, json=payload, headers=headers, timeout=TIMEOUT_S)
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={"node": node["id"], "endpoint": node["endpoint"],
                    "status": resp.status_code, "body": resp.text},
        )
    try:
        return resp.json()
    except ValueError:
        return resp.text


async def _node_transform(node: dict, resolved_input: Any, ctx: dict) -> Any:
    fn = TRANSFORMS.get(node.get("fn"))
    if fn is None:
        raise HTTPException(400, f"node '{node['id']}': unknown transform fn '{node.get('fn')}'")
    return fn(resolved_input)


async def _node_constant(node: dict, resolved_input: Any, ctx: dict) -> Any:
    return node.get("value")


NODE_TYPES: dict[str, Callable] = {
    "serving_endpoint": _node_serving_endpoint,
    "transform": _node_transform,
    "constant": _node_constant,
}


# --------------------------------------------------------------------------- #
# Transforms — pure functions over {upstream_id: output}. Extend freely.
# --------------------------------------------------------------------------- #
def _number_list(x: Any) -> list | None:
    """Dig a flat list of numbers out of a serving response. Handles both the
    flat `{"predictions": [...]}` shape and the nested `{"predictions":
    {"predictions": [...]}}` shape a custom pyfunc produces. Returns None if the
    payload isn't a numeric list."""
    seen = 0
    while isinstance(x, dict) and "predictions" in x and seen < 5:
        x, seen = x["predictions"], seen + 1
    if isinstance(x, list) and x and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
        return x
    return None


def t_mean_predictions(inputs: dict) -> dict:
    """Elementwise-average the numeric predictions of upstream serving responses.
    Falls back to returning the raw upstream outputs if they don't line up."""
    series = [s for s in (_number_list(v) for v in inputs.values()) if s is not None]
    if len(series) >= 2 and len({len(s) for s in series}) == 1:
        mean = [sum(vals) / len(vals) for vals in zip(*series)]
        return {"predictions": mean}
    return {"inputs": inputs}


def t_merge(inputs: dict) -> dict:
    """Return the upstream outputs unchanged, keyed by node id."""
    return {"inputs": inputs}


TRANSFORMS: dict[str, Callable[[dict], Any]] = {
    "mean_predictions": t_mean_predictions,
    "merge": t_merge,
}


# --------------------------------------------------------------------------- #
# Graph execution
# --------------------------------------------------------------------------- #
def _ancestors(nodes: dict, target: str) -> set[str]:
    seen, stack = set(), [target]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        if nid not in nodes:
            raise HTTPException(400, f"graph references unknown node '{nid}'")
        seen.add(nid)
        stack.extend(nodes[nid].get("inputs", []) or [])
    return seen


def _topo_levels(nodes: dict, subset: set[str]) -> list[list[str]]:
    """Kahn's algorithm over `subset`, grouped into levels. Nodes in the same
    level have no edges between them, so they can run concurrently. Raises on
    cycles."""
    indeg = {n: 0 for n in subset}
    for n in subset:
        for dep in nodes[n].get("inputs", []) or []:
            if dep in subset:
                indeg[n] += 1
    levels, remaining = [], set(subset)
    while remaining:
        ready = sorted(n for n in remaining if indeg[n] == 0)
        if not ready:
            raise HTTPException(400, "graph has a cycle")
        levels.append(ready)
        remaining -= set(ready)
        for n in remaining:
            for dep in nodes[n].get("inputs", []) or []:
                if dep in ready:
                    indeg[n] -= 1
    return levels


async def run_graph(graph_name: str, body: Any, client: httpx.AsyncClient) -> dict:
    cfg = load_config()
    graph = cfg["graphs"].get(graph_name)
    if not graph:
        raise HTTPException(404, f"unknown graph '{graph_name}'; configured: {list(cfg['graphs'])}")
    nodes = cfg["nodes"]
    output = graph["output"]

    levels = _topo_levels(nodes, _ancestors(nodes, output))
    results: dict[str, Any] = {}
    ctx = {"client": client}

    async def run_one(nid: str) -> Any:
        node = nodes[nid]
        handler = NODE_TYPES.get(node.get("type"))
        if handler is None:
            raise HTTPException(400, f"node '{nid}': unknown type '{node.get('type')}'")
        ups = node.get("inputs", []) or []
        resolved = body if not ups else {u: results[u] for u in ups}
        return await handler(node, resolved, ctx)

    # Each level's nodes depend only on earlier levels, so run them concurrently.
    for level in levels:
        outs = await asyncio.gather(*(run_one(nid) for nid in level))
        results.update(dict(zip(level, outs)))

    return {"result": results[output], "nodes": results}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict:
    cfg = load_config()
    return {"nodes": list(cfg["nodes"].values()), "graphs": cfg["graphs"]}


@app.post("/graph/{graph_name}")
async def graph(graph_name: str, request: Request) -> Any:
    return await run_graph(graph_name, await request.json(), request.app.state.client)


@app.post("/invoke/{node_id}")
async def invoke(node_id: str, request: Request) -> Any:
    """Convenience: call a single serving_endpoint node directly."""
    cfg = load_config()
    node = cfg["nodes"].get(node_id)
    if not node:
        raise HTTPException(404, f"unknown node '{node_id}'; configured: {list(cfg['nodes'])}")
    if node.get("type") != "serving_endpoint":
        raise HTTPException(400, f"node '{node_id}' is type '{node.get('type')}', not serving_endpoint")
    return await _node_serving_endpoint(node, await request.json(), {"client": request.app.state.client})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    cfg = load_config()
    nrows = "".join(
        f"<li><code>{nid}</code> <em>({n.get('type')})</em>"
        + (f" &rarr; <code>{n['endpoint']}{n.get('path', '/invocations')}</code>" if n.get("type") == "serving_endpoint" else "")
        + (f" &larr; {', '.join(n['inputs'])}" if n.get("inputs") else "")
        + "</li>"
        for nid, n in cfg["nodes"].items()
    )
    grows = "".join(
        f"<li><code>{g}</code> &rarr; output <code>{spec['output']}</code></li>"
        for g, spec in cfg["graphs"].items()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Serving Endpoint Orchestrator</title>
<style>body{{font-family:system-ui,sans-serif;max-width:780px;margin:3rem auto;padding:0 1rem;line-height:1.55}}
code{{background:#f4f4f5;padding:1px 5px;border-radius:4px}}h2{{margin-top:2rem}}</style></head>
<body>
<h1>Serving Endpoint Orchestrator</h1>
<p>Config-driven compute graph over Databricks model serving endpoints.</p>
<h2>Nodes</h2><ul>{nrows or "<li><em>none</em></li>"}</ul>
<h2>Graphs</h2><ul>{grows or "<li><em>none</em></li>"}</ul>
<h2>Routes</h2><ul>
<li><code>POST /graph/{{name}}</code> — run a compute graph</li>
<li><code>POST /invoke/{{node}}</code> — call one serving_endpoint node</li>
<li><code>GET /config</code> · <code>GET /health</code></li>
</ul>
</body></html>"""
