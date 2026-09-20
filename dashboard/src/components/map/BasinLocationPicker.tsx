import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { Crosshair, RotateCcw } from 'lucide-react';
import { OCEAN_PRESETS } from '@/data/presets';
import { NIO_DOMAIN, clampAndSnapCoordinates } from '@/lib/ocean';

interface BasinLocationPickerProps {
  latitude: number;
  longitude: number;
  onSelectCoordinates: (lat: number, lon: number) => void;
  height?: string;
}

export const BasinLocationPicker: React.FC<BasinLocationPickerProps> = ({
  latitude,
  longitude,
  onSelectCoordinates,
  height = '380px',
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.CircleMarker | null>(null);
  const pulseMarkerRef = useRef<L.CircleMarker | null>(null);
  const onSelectRef = useRef(onSelectCoordinates);
  const initialCoordsRef = useRef({ latitude, longitude });

  useEffect(() => {
    onSelectRef.current = onSelectCoordinates;
  });

  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Centered on the North Indian Ocean basin (Arabian Sea, Bay of Bengal, Equatorial zone)
    const map = L.map(mapContainerRef.current, {
      center: [16.5, 75.0],
      zoom: 4,
      minZoom: 3,
      maxZoom: 8,
      attributionControl: false,
      zoomControl: false,
    });

    // Top-right zoom control
    L.control.zoom({ position: 'topright' }).addTo(map);

    // High quality ESRI Ocean Bathymetric basemap tiles
    L.tileLayer(
      'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 10,
        attribution: 'GEBCO, NOAA, ESRI',
      }
    ).addTo(map);

    // Authoritative NIO domain bounding box: 5°N–30°N, 45°E–105°E
    const bounds: L.LatLngBoundsExpression = [
      [NIO_DOMAIN.latMin, NIO_DOMAIN.lonMin],
      [NIO_DOMAIN.latMax, NIO_DOMAIN.lonMax],
    ];
    L.rectangle(bounds, {
      color: '#3b49df',
      weight: 1.5,
      dashArray: '4, 4',
      fill: true,
      fillColor: '#3b49df',
      fillOpacity: 0.03,
    }).addTo(map);

    // Station presets plotted with subtle amber markers
    OCEAN_PRESETS.forEach((preset) => {
      const pMarker = L.circleMarker([preset.latitude, preset.longitude], {
        radius: 4.5,
        color: '#d97706',
        fillColor: '#d97706',
        fillOpacity: 0.9,
        weight: 1.5,
      }).addTo(map);

      pMarker.bindTooltip(
        `<div style="font-family: var(--font-sans, system-ui); font-size: 11px; padding: 2px;">
           <b style="color: #0d253d;">${preset.name}</b><br/>
           <span style="color: #64748d; font-family: var(--font-mono, monospace);">${preset.latitude.toFixed(2)}°N, ${preset.longitude.toFixed(2)}°E</span>
         </div>`,
        { direction: 'top', className: 'ocean-map-tooltip' }
      );

      pMarker.on('click', () => {
        onSelectRef.current(preset.latitude, preset.longitude);
      });
    });

    // Outer halo marker for selected coordinates
    const haloMarker = L.circleMarker(
      [initialCoordsRef.current.latitude, initialCoordsRef.current.longitude],
      {
        radius: 12,
        color: '#533afd',
        fillColor: '#533afd',
        fillOpacity: 0.15,
        weight: 1,
        dashArray: '2, 2',
      }
    ).addTo(map);
    pulseMarkerRef.current = haloMarker;

    // Focal target station marker
    const selMarker = L.circleMarker(
      [initialCoordsRef.current.latitude, initialCoordsRef.current.longitude],
      {
        radius: 6,
        color: '#ffffff',
        fillColor: '#3b49df',
        fillOpacity: 0.95,
        weight: 2,
      }
    ).addTo(map);
    markerRef.current = selMarker;

    // Click on basin to clamp and snap to 0.25° grid
    map.on('click', (e: L.LeafletMouseEvent) => {
      const { latitude: snappedLat, longitude: snappedLon } = clampAndSnapCoordinates(
        e.latlng.lat,
        e.latlng.lng
      );
      onSelectRef.current(snappedLat, snappedLon);
    });

    mapInstanceRef.current = map;

    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => {
      clearTimeout(timer);
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update target marker position when coordinates change
  useEffect(() => {
    if (markerRef.current && pulseMarkerRef.current) {
      markerRef.current.setLatLng([latitude, longitude]);
      pulseMarkerRef.current.setLatLng([latitude, longitude]);
    }
  }, [latitude, longitude]);

  const handleResetView = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([16.5, 75.0], 4);
    }
  };

  return (
    <div
      className="relative isolate w-full rounded-xl overflow-hidden border border-[#e3e8ee] bg-white shadow-xs"
      style={{ height }}
      role="region"
      aria-label="North Indian Ocean Basin Geographic Selector"
    >
      {/* Selected Coordinates Overlay */}
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2 px-2.5 py-1 rounded-md bg-white/95 backdrop-blur-md border border-[#e3e8ee] text-sm font-mono text-[#0d253d] pointer-events-none shadow-xs">
        <Crosshair className="size-3 text-[#533afd]" />
        <span className="tabular-nums font-medium">
          {latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E
        </span>
      </div>

      {/* Reset Map View Button */}
      <button
        type="button"
        onClick={handleResetView}
        title="Reset map view to North Indian Ocean domain"
        className="absolute top-3 right-12 z-20 p-1.5 rounded-md bg-white/95 hover:bg-[#f6f9fc] border border-[#e3e8ee] text-[#64748d] hover:text-[#0d253d] transition-colors cursor-pointer shadow-xs"
        aria-label="Reset basin view"
      >
        <RotateCcw className="size-3.5" />
      </button>

      {/* Leaflet Map Canvas */}
      <div className="w-full h-full" ref={mapContainerRef} />

      {/* Domain Footnote */}
      <div className="absolute bottom-3 left-3 z-20 text-[11px] font-mono text-[#64748d] bg-white/95 backdrop-blur-md px-2.5 py-0.5 rounded-md border border-[#e3e8ee] pointer-events-none shadow-xs">
        0.25° grid · 5°–30°N, 45°–105°E
      </div>
    </div>
  );
};
