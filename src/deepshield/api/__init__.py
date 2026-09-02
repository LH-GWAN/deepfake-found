"""REST API layer."""

from __future__ import annotations

from typing import Any

from deepshield.api.routes import ROUTES

__all__ = ["ROUTES", "create_app"]


def create_app(config: Any = None) -> Any:
    """Lazily build the FastAPI app so importing the package stays dependency-free."""
    from deepshield.api.app import create_app as _create_app

    return _create_app(config)
