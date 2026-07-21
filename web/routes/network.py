"""Clinical-trial network graph API.

``GET /api/network`` returns nodes and edges for a graph whose vertices are
companies (tickers) and clinical trials (NCT IDs), connected by:

  * ``trial``     edges : ticker -> its NCT trials
  * ``sponsor``   edges : two tickers sharing a non-lead sponsor on a trial
  * ``target``    edges : two tickers sharing the same lead-phase / MoA group

The graph is built with ``networkx`` when available (for richer metrics);
otherwise it is built by hand. Either way the JSON shape is identical.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Query

from .. import data_loader as dl

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/network", tags=["network"])

try:
    import networkx as nx  # type: ignore

    _HAS_NX = True
except ImportError:  # networkx optional
    nx = None  # type: ignore
    _HAS_NX = False
    logger.info("networkx not installed; falling back to manual graph building.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shared_sponsor_edges(
    ticker_trials: Dict[str, List[Dict[str, str]]],
    aact_sponsors: Dict[str, List[str]],
) -> List[Tuple[str, str, str, str]]:
    """Find (ticker_a, ticker_b, sponsor, edge_type='sponsor') triples.

    Two tickers are linked when an external (non-lead) sponsor appears on a
    trial of *both* tickers — interpreted as collaborative signalling.
    """
    # Map sponsor -> set of tickers that touch it.
    sponsor_to_tickers: Dict[str, Set[str]] = defaultdict(set)
    for ticker, trials in ticker_trials.items():
        own = {(t.get("sponsor_name_at_map_time") or "").lower() for t in trials if t.get("sponsor_name_at_map_time")}
        for t in trials:
            nct = t.get("nct_id", "")
            for sp in aact_sponsors.get(nct, []):
                if sp.lower() in own:
                    continue
                sponsor_to_tickers[sp].add(ticker)

    edges: List[Tuple[str, str, str, str]] = []
    for sponsor, tickers in sponsor_to_tickers.items():
        if len(tickers) < 2:
            continue
        ts = sorted(tickers)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                edges.append((ts[i], ts[j], sponsor, "sponsor"))
    return edges


def _target_edges(cells: List[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
    """Connect tickers sharing the same lead phase ("target" edge).

    Companies with the same pipeline stage bucket and tissue are assumed to
    be pursuing comparable targets / modalities.
    """
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for c in cells:
        phase = (c.get("nucleus") or {}).get("lead_phase") or "none"
        tissue = c.get("tissue") or "Unknown"
        groups[(phase, tissue)].append(c["ticker"])

    edges: List[Tuple[str, str, str, str]] = []
    for (phase, tissue), tickers in groups.items():
        if len(tickers) < 2:
            continue
        ts = sorted(tickers)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                label = f"{phase} / {tissue}"
                edges.append((ts[i], ts[j], label, "target"))
    return edges


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(snapshot: Optional[str] = None) -> Dict[str, Any]:
    """Construct the network graph payload."""
    cells = dl.build_all_cells(snapshot)
    ticker_trials = dl.load_trial_mapping()
    aact_sponsors = dl.load_aact_sponsors()
    aact_studies = dl.load_aact_studies()

    ticker_index: Dict[str, Dict[str, Any]] = {c["ticker"]: c for c in cells}

    # ---- Build nodes ----
    company_nodes: List[Dict[str, Any]] = []
    trial_nodes: List[Dict[str, Any]] = []
    seen_ncts: Set[str] = set()
    trial_edges: List[Tuple[str, str, str, str]] = []

    for ticker, trials in ticker_trials.items():
        cell = ticker_index.get(ticker)
        company_nodes.append(
            {
                "id": ticker,
                "type": "company",
                "label": (cell.get("name") if cell else ticker) or ticker,
                "tissue": (cell.get("tissue") if cell else "Diversified") or "Diversified",
                "composite_score": (cell.get("composite_score") if cell else None),
                "trial_count": len(trials),
                "catalyst_count": (cell.get("receptors", {}).get("catalyst_count") if cell else 0),
            }
        )
        for t in trials:
            nct = t.get("nct_id", "")
            if not nct or nct in seen_ncts:
                # still record the edge if the trial node already exists
                trial_edges.append((ticker, nct, nct, "trial"))
                continue
            seen_ncts.add(nct)
            study = aact_studies.get(nct, {})
            trial_nodes.append(
                {
                    "id": nct,
                    "type": "trial",
                    "label": nct,
                    "phase": study.get("phase", "Unknown"),
                    "status": study.get("overall_status", "Unknown"),
                    "primary_completion_date": study.get("primary_completion_date", ""),
                    "sponsors": aact_sponsors.get(nct, []),
                }
            )
            trial_edges.append((ticker, nct, nct, "trial"))

    # ---- Build inter-company edges ----
    sponsor_edges = _shared_sponsor_edges(ticker_trials, aact_sponsors)
    target_edges = _target_edges(cells)

    # ---- Assemble edges ----
    all_edges: List[Dict[str, Any]] = []
    for src, dst, label, etype in trial_edges:
        all_edges.append({"source": src, "target": dst, "type": etype, "label": label})
    for src, dst, label, etype in sponsor_edges:
        all_edges.append({"source": src, "target": dst, "type": etype, "label": label})
    for src, dst, label, etype in target_edges:
        all_edges.append({"source": src, "target": dst, "type": etype, "label": label})

    # ---- Metrics via networkx if present, else hand-rolled ----
    metrics: Dict[str, Any] = {}
    if _HAS_NX:
        G = nx.Graph()
        for n in company_nodes:
            G.add_node(n["id"], **n)
        for n in trial_nodes:
            G.add_node(n["id"], **n)
        for e in all_edges:
            G.add_edge(e["source"], e["target"], **{k: v for k, v in e.items() if k not in ("source", "target")})
        metrics = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "connected_components": nx.number_connected_components(G),
            "density": nx.density(G),
            "company_degrees": {n["id"]: G.degree(n["id"]) for n in company_nodes},
            "most_connected_company": max(
                ((n["id"], G.degree(n["id"])) for n in company_nodes),
                key=lambda kv: kv[1],
                default=(None, 0),
            ),
        }
    else:
        # Manual degree computation.
        degree: Dict[str, int] = defaultdict(int)
        for e in all_edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1
        node_ids = {n["id"] for n in company_nodes} | {n["id"] for n in trial_nodes}
        # Connected components via BFS over an adjacency map.
        adj: Dict[str, Set[str]] = defaultdict(set)
        for e in all_edges:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])
        visited: Set[str] = set()
        components = 0
        for nid in node_ids:
            if nid in visited:
                continue
            components += 1
            stack = [nid]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                stack.extend(adj[cur] - visited)
        n_nodes = len(node_ids)
        n_edges = len(all_edges)
        density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
        metrics = {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "connected_components": components,
            "density": density,
            "company_degrees": {n["id"]: degree[n["id"]] for n in company_nodes},
            "most_connected_company": max(
                ((n["id"], degree[n["id"]]) for n in company_nodes),
                key=lambda kv: kv[1],
                default=(None, 0),
            ),
        }

    return {
        "snapshot": snapshot or (dl.load_snapshot_dates()[-1] if dl.load_snapshot_dates() else None),
        "nodes": {
            "companies": company_nodes,
            "trials": trial_nodes,
            "total": len(company_nodes) + len(trial_nodes),
        },
        "edges": {
            "total": len(all_edges),
            "by_type": {
                "trial": sum(1 for e in all_edges if e["type"] == "trial"),
                "sponsor": sum(1 for e in all_edges if e["type"] == "sponsor"),
                "target": sum(1 for e in all_edges if e["type"] == "target"),
            },
            "list": all_edges,
        },
        "metrics": metrics,
        "engine": "networkx" if _HAS_NX else "manual",
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("")
def get_network(snapshot: Optional[str] = Query(default=None)):
    """Return the full graph (nodes + edges + metrics)."""
    return build_graph(snapshot)


@router.get("/companies")
def get_company_graph(snapshot: Optional[str] = Query(default=None)):
    """Company-only subgraph (inter-company sponsor/target edges)."""
    g = build_graph(snapshot)
    company_ids = {n["id"] for n in g["nodes"]["companies"]}
    edges = [e for e in g["edges"]["list"] if e["source"] in company_ids and e["target"] in company_ids]
    return {
        "snapshot": g["snapshot"],
        "nodes": g["nodes"]["companies"],
        "edges": {"total": len(edges), "list": edges},
        "metrics": {
            "node_count": len(company_ids),
            "edge_count": len(edges),
            "engine": g["metrics"].get("engine", g["engine"]),
        },
    }
