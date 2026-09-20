/**
 * TypeScript definitions mirroring backend OceanEmbed FastAPI Pydantic models.
 */

export interface SurfaceContext {
  sst_c: number;
  sss_psu: number;
  ssh_m: number;
  current_u_ms: number;
  current_v_ms: number;
  wind_u_ms: number;
  wind_v_ms: number;
}

export interface ArgoObservation {
  float_id: string;
  date: string;
  latitude: number;
  longitude: number;
  distance_km: number;
  depths_m: number[];
  temperature_c: number[];
  rmse?: number | null;
  mae?: number | null;
  bias?: number | null;
  is_mock: boolean;
}

export interface EmbeddingCoordinates {
  pca_1: number;
  pca_2: number;
  regime_label: string;
  vector_dim: number;
}

export interface EmbeddingScatterPoint {
  id: string;
  pca_1: number;
  pca_2: number;
  regime: string;
  region: string;
  season: string;
  latitude: number;
  longitude: number;
}

export interface EmbeddingScatterResponse {
  points: EmbeddingScatterPoint[];
  regimes: string[];
  total_points: number;
  is_mock: boolean;
}

export interface ModelMetadata {
  name: string;
  version: string;
  provider_type: string;
  checkpoint_hash?: string | null;
  inference_time_ms: number;
}

export interface ReconstructionRequest {
  date: string;
  latitude: number;
  longitude: number;
}

export interface ReconstructionResponse {
  request_id: string;
  date: string;
  latitude: number;
  longitude: number;
  depths_m: number[];
  temperature_c: number[];
  surface_context: SurfaceContext;
  d26_depth_m?: number | null;
  mixed_layer_depth_m?: number | null;
  argo_comparison?: ArgoObservation | null;
  embedding: EmbeddingCoordinates;
  model: ModelMetadata;
  is_mock: boolean;
  provenance: string;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  project: string;
  version: string;
  active_provider: string;
  is_mock: boolean;
  domain: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  depths_m: number[];
}

export interface OceanPreset {
  name: string;
  description: string;
  latitude: number;
  longitude: number;
  date: string;
  region: string;
}
