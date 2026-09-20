"""Real ML Inference Provider for OceanEmbed.

Connects the separately developed PyTorch model checkpoint / ONNX runtime
to the FastAPI inference route.
"""

import os
from typing import Optional
from api.schemas.reconstruction import ReconstructionRequest, ReconstructionResponse
from api.schemas.embedding import EmbeddingScatterResponse
from api.schemas.argo import ArgoObservation
from ml_interface.base_provider import AbstractOceanEmbedProvider
from ml_interface.model_loader import load_pytorch_checkpoint, detect_optimal_device


class RealOceanEmbedProvider(AbstractOceanEmbedProvider):
    """Executes live PyTorch / ONNX model inference."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._provider_name = "pytorch_checkpoint"
        self._checkpoint_path = checkpoint_path or os.getenv(
            "OCEANEMBED_CHECKPOINT_PATH", "checkpoints/oceanembed_weights.pt"
        )
        self._device = detect_optimal_device()
        self._model, self._checkpoint_hash = load_pytorch_checkpoint(self._checkpoint_path)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    def predict_profile(self, request: ReconstructionRequest) -> ReconstructionResponse:
        if not self.is_model_loaded:
            raise RuntimeError(
                f"Real OceanEmbed model checkpoint not found at '{self._checkpoint_path}'. "
                "The ML model is developed separately. Please supply a valid checkpoint or "
                "switch to the mock provider by setting DEFAULT_PROVIDER=mock."
            )
        # When model checkpoint is plugged in by the ML teammate:
        # 1. Normalize input variables with train-year stats
        # 2. Convert to torch.Tensor [1, 14, 1, 1] or [1, 14]
        # 3. Model forward pass -> [1, 15] temperatures and [1, 128] embedding
        # 4. Map embedding via fitted PCA projector
        raise NotImplementedError("ML teammate model forward pass boundary.")

    def get_embedding_scatter(self) -> EmbeddingScatterResponse:
        raise NotImplementedError("Live PCA projection scatter points.")

    def find_nearby_argo(
        self, date_str: str, latitude: float, longitude: float
    ) -> Optional[ArgoObservation]:
        raise NotImplementedError("Live in-situ Argo database lookup.")
