import type { EmbeddingScatterPoint } from '../types/api';

/**
 * North Indian Ocean Domain Boundaries & Resolution
 * Authoritative: 5°N–30°N, 45°E–105°E, 0.25° grid
 */
export const NIO_DOMAIN = {
  latMin: 5.0,
  latMax: 30.0,
  lonMin: 45.0,
  lonMax: 105.0,
  gridResolution: 0.25,
} as const;

/**
 * 15 Authoritative Standard Depths (meters) for OceanEmbed subsurface thermal reconstruction
 */
export const STANDARD_DEPTHS_M = [
  0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000,
] as const;

/**
 * Date range bounds for observation archive
 */
export const OBSERVATION_DATE_RANGE = {
  min: '2015-01-01',
  max: '2024-12-31',
} as const;

/**
 * Validates whether given coordinates fall within the North Indian Ocean domain.
 */
export function isWithinNioDomain(latitude: number, longitude: number): boolean {
  return (
    latitude >= NIO_DOMAIN.latMin &&
    latitude <= NIO_DOMAIN.latMax &&
    longitude >= NIO_DOMAIN.lonMin &&
    longitude <= NIO_DOMAIN.lonMax
  );
}

/**
 * Snaps a coordinate value to the nearest 0.25° grid step.
 */
export function snapCoordinateToGrid(value: number, step = NIO_DOMAIN.gridResolution): number {
  const factor = 1 / step;
  return Math.round(value * factor) / factor;
}

/**
 * Clamps coordinates to the NIO domain and snaps them to the 0.25° grid.
 */
export function clampAndSnapCoordinates(
  lat: number,
  lon: number
): { latitude: number; longitude: number } {
  const clampedLat = Math.max(NIO_DOMAIN.latMin, Math.min(NIO_DOMAIN.latMax, lat));
  const clampedLon = Math.max(NIO_DOMAIN.lonMin, Math.min(NIO_DOMAIN.lonMax, lon));
  return {
    latitude: snapCoordinateToGrid(clampedLat),
    longitude: snapCoordinateToGrid(clampedLon),
  };
}

/**
 * Computes speed magnitude and meteorological heading in degrees (0–360°)
 * from orthogonal vector components (u, v).
 */
export function calculateVectorKinematics(
  u: number,
  v: number
): { speed: number; headingDeg: number } {
  const speed = Math.sqrt(u ** 2 + v ** 2);
  const headingDeg = ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
  return { speed, headingDeg };
}

/**
 * Groups latent embedding scatter points by their oceanographic regime.
 * Pure data transformation helper for embedding manifold analysis.
 */
export function groupScatterPointsByRegime(
  points: EmbeddingScatterPoint[]
): Record<string, EmbeddingScatterPoint[]> {
  const grouped: Record<string, EmbeddingScatterPoint[]> = {};
  for (const p of points) {
    if (!grouped[p.regime]) {
      grouped[p.regime] = [];
    }
    grouped[p.regime].push(p);
  }
  return grouped;
}
