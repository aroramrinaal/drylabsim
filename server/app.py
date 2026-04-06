"""FastAPI application for the DryLabSim environment.

Endpoints:
    - POST /reset:  Reset the environment
    - POST /step:   Execute an action
    - GET  /state:  Get current environment state
    - GET  /schema: Get action/observation schemas
    - WS   /ws:     WebSocket endpoint for persistent sessions
    - GET  /        Demo UI
"""

import os
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'"
    ) from e

from fastapi.responses import HTMLResponse, FileResponse

try:
    from ..models import ExperimentAction, ExperimentObservation
    from .drylabsim_environment import BioExperimentEnvironment
except ImportError:  # pragma: no cover - direct module import path
    from models import ExperimentAction, ExperimentObservation
    from server.drylabsim_environment import BioExperimentEnvironment

app = create_app(
    BioExperimentEnvironment,
    ExperimentAction,
    ExperimentObservation,
    env_name="drylabsim",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "4")),
)

# Serve demo UI at root
DEMO_HTML = Path(__file__).resolve().parent.parent / "demo.html"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    if DEMO_HTML.exists():
        return HTMLResponse(content=DEMO_HTML.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>DryLabSim API</h1><p>Visit /docs for API documentation.</p>",
        status_code=200,
    )


@app.get("/assets/{filename}")
async def serve_asset(filename: str):
    file_path = ASSETS_DIR / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return HTMLResponse(content="Not found", status_code=404)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""

    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    resolved_port = args.port
    if resolved_port is None:
        resolved_port = int(os.environ.get("PORT", str(port)))
    uvicorn.run(app, host=args.host, port=resolved_port)


if __name__ == "__main__":
    main()
