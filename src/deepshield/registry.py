"""Component registry that keeps the system independent of any single model.

Every replaceable capability - face detector, aligner, embedder, deepfake
detector, watermark backend - is looked up by name through a registry instead of
being imported directly by pipeline code. Swapping InsightFace for FaceNet, or a
mock detector for a pretrained one, is then a configuration change.

Example:
    >>> registry = ComponentRegistry[str]("demo")
    >>> registry.register("upper", lambda cfg: "UPPER")
    >>> registry.create("upper", None)
    'UPPER'

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar, cast

from deepshield.exceptions import ModelNotAvailableError

T = TypeVar("T")

Factory = Callable[..., Any]


class ComponentRegistry(Generic[T]):
    """Name-to-factory registry for one kind of pluggable component."""

    def __init__(self, kind: str) -> None:
        """Create an empty registry describing components of ``kind``."""
        self.kind = kind
        self._factories: dict[str, Factory] = {}

    def register(self, name: str, factory: Factory, overwrite: bool = False) -> None:
        """Register ``factory`` under ``name``.

        Raises:
            ValueError: If the name is already taken and ``overwrite`` is False.

        """
        key = name.lower()
        if key in self._factories and not overwrite:
            raise ValueError(f"{self.kind} backend '{name}' is already registered")
        self._factories[key] = factory

    def decorator(self, name: str) -> Callable[[Factory], Factory]:
        """Return a decorator that registers the decorated callable as ``name``."""

        def wrapper(factory: Factory) -> Factory:
            self.register(name, factory)
            return factory

        return wrapper

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Instantiate the backend registered as ``name``.

        Raises:
            ModelNotAvailableError: If no backend is registered under that name.

        """
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            available = ", ".join(self.available()) or "none"
            raise ModelNotAvailableError(
                f"unknown {self.kind} backend '{name}'; available backends: {available}"
            )
        return cast(T, factory(*args, **kwargs))

    def available(self) -> list[str]:
        """Return the registered backend names, sorted."""
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        """Return whether a backend name is registered."""
        return isinstance(name, str) and name.lower() in self._factories

    def __iter__(self) -> Iterable[str]:
        """Iterate over registered backend names."""
        return iter(self.available())

    def __len__(self) -> int:
        """Return the number of registered backends."""
        return len(self._factories)
