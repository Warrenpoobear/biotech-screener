"""FastAPI application for the Biotech Screener Cell Atlas.

Run from the project root::

    uvicorn web.app:app --reload --port 8000

Then open http://localhost:8000/ for the interactive dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data_loader as dl
from .routes import atlas, backtest, network

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("web.app")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Biotech Screener — Cell Atlas",
    description=(
        "Interactive cell-atlas visualisation of the biotech investment "
        "universe. Each company is a 'cell' with membrane, nucleus, "
        "organelles, receptors and signalling connections."
    ),
    version="1.0.0",
)

# ---- CORS (allow local dev + any origin for dashboards) ----
app.add_middleware(
    CORSMiddleware,
    # Local single-user dashboard: scope to localhost origins. The previous
    # wildcard + allow_credentials=True combination is rejected by browsers
    # per the CORS spec, and no cookies/auth are in use here.
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routers ----
app.include_router(atlas.router)
app.include_router(backtest.router)
app.include_router(network.router)

# ---- Static files (JS/CSS assets) ----
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Root + convenience routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    """Serve the single-page dashboard at the root URL."""
    return FileResponse(STATIC_DIR / "atlas.html")


@app.get("/api/health", tags=["meta"])
def health():
    """Liveness + data-availability probe."""
    universe = dl.load_universe()
    snapshots = dl.load_snapshot_dates()
    return {
        "status": "ok",
        "universe_tickers": len(universe),
        "snapshots": snapshots,
        "repo_root": str(dl.REPO_ROOT),
        "snapshots_dir": str(dl.snapshots_dir()),
    }


@app.post("/api/reload", tags=["meta"])
def reload_cache():
    """Clear the in-memory cache so subsequent requests re-read disk."""
    dl.clear_cache()
    return {"status": "reloaded", "universe_tickers": len(dl.load_universe())}
