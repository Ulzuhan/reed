"""Reed — self-hosted RAG service. Reed has read your documents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reed")
except PackageNotFoundError:  # Source tree imported without being installed.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
