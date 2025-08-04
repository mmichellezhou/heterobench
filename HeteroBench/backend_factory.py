from typing import Dict, Type
from backend import Backend
from backends.cpu.cpu_backend import CPUBackend


class BackendFactory:
    """Factory for creating backend instances."""

    _backends: Dict[str, Type[Backend]] = {
        "cpu": CPUBackend,
    }

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]):
        """Register a new backend."""
        cls._backends[name] = backend_class

    @classmethod
    def create_backend(cls, name: str) -> Backend:
        """Create a backend instance by name."""
        if name not in cls._backends:
            raise ValueError(
                f"Unknown backend: {name}. Available backends: {list(cls._backends.keys())}"
            )

        return cls._backends[name]()

    @classmethod
    def get_available_backends(cls) -> list:
        """Get list of available backend names."""
        return list(cls._backends.keys())
