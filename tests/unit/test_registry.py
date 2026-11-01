"""Component registry behaviour."""

from __future__ import annotations

import pytest

from deepshield.exceptions import ModelNotAvailableError
from deepshield.registry import ComponentRegistry


def test_register_and_create() -> None:
    registry: ComponentRegistry[str] = ComponentRegistry("thing")
    registry.register("alpha", lambda: "A")
    assert registry.create("alpha") == "A"


def test_lookup_is_case_insensitive() -> None:
    registry: ComponentRegistry[str] = ComponentRegistry("thing")
    registry.register("Alpha", lambda: "A")
    assert registry.create("ALPHA") == "A"
    assert "alpha" in registry


def test_duplicate_registration_is_rejected() -> None:
    registry: ComponentRegistry[str] = ComponentRegistry("thing")
    registry.register("alpha", lambda: "A")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("alpha", lambda: "B")


def test_overwrite_is_explicit() -> None:
    registry: ComponentRegistry[str] = ComponentRegistry("thing")
    registry.register("alpha", lambda: "A")
    registry.register("alpha", lambda: "B", overwrite=True)
    assert registry.create("alpha") == "B"


def test_unknown_backend_lists_alternatives() -> None:
    registry: ComponentRegistry[str] = ComponentRegistry("face embedder")
    registry.register("mock", lambda: "M")
    with pytest.raises(ModelNotAvailableError) as excinfo:
        registry.create("insightface")
    message = str(excinfo.value)
    assert "insightface" in message
    assert "mock" in message


def test_decorator_registers_factory() -> None:
    registry: ComponentRegistry[str] = ComponentRegistry("thing")

    @registry.decorator("beta")
    def make() -> str:
        return "B"

    assert registry.create("beta") == "B"
    assert make() == "B"


def test_available_is_sorted_and_len_matches() -> None:
    registry: ComponentRegistry[str] = ComponentRegistry("thing")
    registry.register("zeta", lambda: "Z")
    registry.register("alpha", lambda: "A")
    assert registry.available() == ["alpha", "zeta"]
    assert len(registry) == 2
