"""Centralized location for all configuration settings in ARIEL."""

# Third-party libraries
from pydantic_settings import BaseSettings


class ArielConfig(BaseSettings):
    """Global ARIEL settings."""

    verbose: bool = True
