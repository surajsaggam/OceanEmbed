/**
 * Typed API Client for OceanEmbed FastAPI backend.
 */

import type {
  HealthResponse,
  ReconstructionRequest,
  ReconstructionResponse,
  EmbeddingScatterResponse,
  ArgoObservation,
  ReconstructionHistoryItem,
  TransectRequest,
  TransectResponse,
  DepartureRequest,
  DepartureResponse,
  DepthDepartureMetrics,
} from '../types/api';


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error (${status}): ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = 'Unknown server error';
    try {
      const data = await response.json();
      if (typeof data.detail === 'string') {
        errorDetail = data.detail;
      } else if (Array.isArray(data.detail)) {
        errorDetail = data.detail.map((e: any) => `${e.loc?.join('.')} ${e.msg}`).join('; ');
      } else {
        errorDetail = JSON.stringify(data);
      }
    } catch {
      errorDetail = response.statusText;
    }
    throw new ApiError(response.status, errorDetail);
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse<HealthResponse>(response);
}

export async function reconstructProfile(
  request: ReconstructionRequest
): Promise<ReconstructionResponse> {
  const response = await fetch(`${API_BASE_URL}/reconstruct`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(request),
  });
  return handleResponse<ReconstructionResponse>(response);
}

export async function fetchEmbeddingScatter(): Promise<EmbeddingScatterResponse> {
  const response = await fetch(`${API_BASE_URL}/embedding`);
  return handleResponse<EmbeddingScatterResponse>(response);
}

export async function fetchNearbyArgo(
  date: string,
  latitude: number,
  longitude: number
): Promise<ArgoObservation | null> {
  const params = new URLSearchParams({
    date,
    latitude: latitude.toString(),
    longitude: longitude.toString(),
  });
  const response = await fetch(`${API_BASE_URL}/argo/nearby?${params.toString()}`);
  return handleResponse<ArgoObservation | null>(response);
}

export async function fetchReconstructionHistory(
  limit: number = 20
): Promise<ReconstructionHistoryItem[]> {
  const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`);
  return handleResponse<ReconstructionHistoryItem[]>(response);
}

export async function downloadReconstructionPdf(
  reconstruction: ReconstructionResponse,
  transect?: TransectResponse | null
): Promise<void> {
  const payload = {
    ...reconstruction,
    transect: transect || undefined,
  };
  const response = await fetch(`${API_BASE_URL}/report/pdf`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/pdf',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorDetail = 'Failed to generate PDF report';
    try {
      const data = await response.json();
      if (data.detail) errorDetail = data.detail;
    } catch {
      errorDetail = response.statusText;
    }
    throw new ApiError(response.status, errorDetail);
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = `OceanIQ_Report_${reconstruction.date}_${reconstruction.latitude.toFixed(2)}N_${reconstruction.longitude.toFixed(2)}E.pdf`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

export async function fetchTransect(
  request: TransectRequest
): Promise<TransectResponse> {
  const response = await fetch(`${API_BASE_URL}/transect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(request),
  });
  return handleResponse<TransectResponse>(response);
}

export async function fetchDeparture(
  request: DepartureRequest
): Promise<DepartureResponse> {
  const response = await fetch(`${API_BASE_URL}/departure`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(request),
  });
  return handleResponse<DepartureResponse>(response);
}

export async function fetchSkillProfile(
  date: string = '2019-01-01'
): Promise<DepthDepartureMetrics[]> {
  const response = await fetch(
    `${API_BASE_URL}/departure/skill-profile?date=${encodeURIComponent(date)}`
  );
  return handleResponse<DepthDepartureMetrics[]>(response);
}




