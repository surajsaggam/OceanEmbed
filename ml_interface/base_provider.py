"""Abstract Base Class for OceanEmbed inference providers.

Both the MockInferenceProvider (used for UI development & testing)
and the RealOceanEmbedProvider (used when the ML teammate supplies the checkpoint)
must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from api.schemas.reconstruction import ReconstructionRequest, ReconstructionResponse
from api.schemas.embedding import EmbeddingScatterResponse
from api.schemas.argo import ArgoObservation
from api.schemas.transect import TransectRequest, TransectResponse
from api.schemas.departure import DepartureRequest, DepartureResponse


class AbstractOceanEmbedProvider(ABC):
    """Contract for subsurface ocean temperature reconstruction providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'mock_climatology' or 'pytorch_checkpoint')."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Boolean flag indicating whether data is synthetic/mock or live ML inference."""
        pass

    @abstractmethod
    def predict_profile(self, request: ReconstructionRequest) -> ReconstructionResponse:
        """Reconstruct subsurface temperature across the 15 standard depths.
        
        Args:
            request: Validated date and coordinates within the North Indian Ocean.
            
        Returns:
            ReconstructionResponse containing the 15-depth temperature profile,
            surface context, D26 isotherm, Argo comparison if available, and provenance.
        """
        pass

    @abstractmethod
    def get_embedding_scatter(self) -> EmbeddingScatterResponse:
        """Fetch 2D PCA/UMAP background points representing oceanographic regimes.
        
        Returns:
            EmbeddingScatterResponse containing historical/reference points across
            Arabian Sea, Bay of Bengal, and Equatorial Indian Ocean.
        """
        pass

    @abstractmethod
    def find_nearby_argo(
        self, date_str: str, latitude: float, longitude: float
    ) -> Optional[ArgoObservation]:
        """Find an independent in-situ Argo float observation collocated in time and space.
        
        Args:
            date_str: ISO date string (YYYY-MM-DD).
            latitude: Query latitude (°N).
            longitude: Query longitude (°E).
            
        Returns:
            ArgoObservation if a float is found within tolerance, otherwise None.
        """
        pass

    @abstractmethod
    def predict_transect(self, request: TransectRequest) -> TransectResponse:
        """Reconstruct subsurface temperature along a geographic vertical transect.
        
        Args:
            request: TransectRequest containing date, waypoint coordinates, and sample resolution.
            
        Returns:
            TransectResponse containing discrete stations with 15 standard depth temperatures.
        """
        pass

    @abstractmethod
    def get_reconstruction_departure(self, request: DepartureRequest) -> DepartureResponse:
        """Calculate and return reconstruction departure relative to GLORYS12V1 reanalysis reference.
        
        Args:
            request: DepartureRequest containing date, depth level, and optional coordinates.
            
        Returns:
            DepartureResponse containing depth statistics and 2D spatial departure field.
        """
        pass


