"""Every module must import cleanly without optional heavy dependencies.

The MVP install carries only numpy, Pillow, PyYAML and pydantic. Importing the
package must never pull in torch, InsightFace or FastAPI, so a minimal install
can still run the CLI.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

import deepshield

HEAVY_MODULES = ("torch", "insightface", "fastapi", "cv2", "sqlalchemy", "onnxruntime")


def _module_names() -> list[str]:
    return [
        name
        for _, name, _ in pkgutil.walk_packages(deepshield.__path__, f"{deepshield.__name__}.")
    ]


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_importing_package_does_not_pull_heavy_dependencies() -> None:
    for name in list(sys.modules):
        if name.startswith("deepshield"):
            del sys.modules[name]
    already_loaded = {name for name in HEAVY_MODULES if name in sys.modules}
    importlib.import_module("deepshield")
    newly_loaded = {name for name in HEAVY_MODULES if name in sys.modules} - already_loaded
    assert newly_loaded == set()


def test_public_exceptions_are_exported() -> None:
    for name in ("DeepShieldError", "ConfigurationError", "NotImplementedInPhaseError"):
        assert hasattr(deepshield, name)


def test_api_module_imports_without_fastapi_installed() -> None:
    """The API module must import even on a minimal install, and say so if unusable."""
    from deepshield.api import app as api_module
    from deepshield.exceptions import ModelNotAvailableError

    if api_module.FASTAPI_AVAILABLE:
        assert api_module.create_app() is not None
    else:
        with pytest.raises(ModelNotAvailableError, match="FastAPI is not installed"):
            api_module.create_app()


def test_planned_routes_are_declared() -> None:
    from deepshield.api.routes import ROUTES

    paths = {path for _, path, _ in ROUTES}
    assert {"/identity/enroll", "/analyze/image", "/analyze/video", "/health"} <= paths
