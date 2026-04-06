"""Route tests for the FastAPI app wiring."""

from importlib import import_module
import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient


def _load_app(enable_web_interface: bool):
    module_name = "server.app"
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    sys.modules.pop(module_name, None)
    os.environ["ENABLE_WEB_INTERFACE"] = "true" if enable_web_interface else "false"
    return import_module(module_name).app


class TestAppRoutes:
    def test_root_redirects_to_demo_when_web_interface_enabled(self):
        client = TestClient(_load_app(enable_web_interface=True))

        response = client.get("/", follow_redirects=False)

        assert response.status_code == 308
        assert response.headers["location"] == "/demo"

    def test_demo_and_core_routes_are_available(self):
        client = TestClient(_load_app(enable_web_interface=False))

        demo_response = client.get("/demo")
        asset_response = client.get("/demo/assets/favicon.ico")
        reset_response = client.post("/reset")
        step_response = client.post("/step")
        state_response = client.get("/state")

        assert demo_response.status_code == 200
        assert asset_response.status_code == 200
        assert reset_response.status_code == 200
        assert step_response.status_code == 422
        assert state_response.status_code == 200
