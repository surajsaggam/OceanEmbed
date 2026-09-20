import { useState, useEffect, useCallback, useRef } from 'react';
import {
  fetchHealth,
  reconstructProfile,
  fetchEmbeddingScatter,
} from '../services/api';
import type {
  ReconstructionResponse,
  EmbeddingScatterResponse,
  HealthResponse,
  OceanPreset,
} from '../types/api';

export interface UseOceanEmbedOptions {
  initialDate?: string;
  initialLatitude?: number;
  initialLongitude?: number;
  autoReconstructOnMount?: boolean;
}

export function useOceanEmbed(options: UseOceanEmbedOptions = {}) {
  const {
    initialDate = '2023-06-15',
    initialLatitude = 18.5,
    initialLongitude = 88.25,
    autoReconstructOnMount = true,
  } = options;

  const [date, setDate] = useState<string>(initialDate);
  const [latitude, setLatitude] = useState<number>(initialLatitude);
  const [longitude, setLongitude] = useState<number>(initialLongitude);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [reconstruction, setReconstruction] = useState<ReconstructionResponse | null>(null);
  const [scatterData, setScatterData] = useState<EmbeddingScatterResponse | null>(null);

  // Initialize server health status and latent manifold data
  useEffect(() => {
    let isMounted = true;

    async function init() {
      try {
        const h = await fetchHealth();
        if (isMounted) setHealth(h);
      } catch (err: unknown) {
        console.error('Health check failed:', err);
      }

      try {
        const scat = await fetchEmbeddingScatter();
        if (isMounted) setScatterData(scat);
      } catch (err: unknown) {
        console.error('Embedding scatter fetch failed:', err);
      }
    }

    init();

    return () => {
      isMounted = false;
    };
  }, []);

  const executeReconstruction = useCallback(
    async (targetLat = latitude, targetLon = longitude, targetDate = date) => {
      setLoading(true);
      setError(null);
      try {
        const resp = await reconstructProfile({
          date: targetDate,
          latitude: targetLat,
          longitude: targetLon,
        });
        setReconstruction(resp);
        return resp;
      } catch (err: any) {
        console.error('Reconstruction failed:', err);
        const errMsg = err?.detail || err?.message || 'Failed to reconstruct subsurface profile';
        setError(errMsg);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [latitude, longitude, date]
  );

  // Initial station reconstruction on mount
  const initialRunRef = useRef(false);
  useEffect(() => {
    if (autoReconstructOnMount && !initialRunRef.current) {
      initialRunRef.current = true;
      executeReconstruction(initialLatitude, initialLongitude, initialDate);
    }
  }, [autoReconstructOnMount, executeReconstruction, initialLatitude, initialLongitude, initialDate]);

  const selectPreset = useCallback(
    (preset: OceanPreset) => {
      setLatitude(preset.latitude);
      setLongitude(preset.longitude);
      setDate(preset.date);
      return executeReconstruction(preset.latitude, preset.longitude, preset.date);
    },
    [executeReconstruction]
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    date,
    setDate,
    latitude,
    setLatitude,
    longitude,
    setLongitude,
    loading,
    error,
    clearError,
    health,
    reconstruction,
    scatterData,
    executeReconstruction,
    selectPreset,
  };
}
