"""Inference service dispatcher for OceanEmbed.

Manages active provider instance (MockInferenceProvider or RealOceanEmbedProvider)
and exposes dependency-injected provider accessor for FastAPI routes.
"""

import os
from typing import Optional
from api.config import settings
from ml_interface.base_provider import AbstractOceanEmbedProvider
from api.services.mock_provider import MockInferenceProvider
from api.services.real_provider import RealOceanEmbedProvider


class InferenceService:
    """Singleton dispatcher managing the active ocean temperature inference provider."""

    def __init__(self, provider_mode: Optional[str] = None) -> None:
        mode = provider_mode or os.getenv("OCEANEMBED_PROVIDER", settings.DEFAULT_PROVIDER)
        self.mode = mode.lower()
        self._mock_provider = MockInferenceProvider()
        self._real_provider = RealOceanEmbedProvider()

    def get_provider(self) -> AbstractOceanEmbedProvider:
        if self.mode == "real" and self._real_provider.is_model_loaded:
            return self._real_provider
        return self._mock_provider

    def set_mode(self, mode: str) -> None:
        if mode not in ["mock", "real"]:
            raise ValueError("Mode must be either 'mock' or 'real'")
        self.mode = mode


# Global singleton instance
_inference_service = InferenceService()


def get_inference_service() -> InferenceService:
    """Dependency injector for FastAPI routes."""
    return _inference_service
