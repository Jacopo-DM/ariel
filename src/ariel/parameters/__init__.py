# ruff: file-ignore[non-empty-init-module]  # package entry point: runs setup code before the re-exports (see CODEMAP.md)
"""Centralized location for all configuration settings in ARIEL."""

# Third-party libraries
from pydantic_settings import BaseSettings


class ArielConfig(BaseSettings):
    """Global ARIEL settings."""

    verbose: bool = True
