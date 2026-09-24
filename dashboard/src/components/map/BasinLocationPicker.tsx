import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { Crosshair, RotateCcw, Route, MapPin, X } from 'lucide-react';
import { OCEAN_PRESETS } from '@/data/presets';
import { NIO_DOMAIN, clampAndSnapCoordinates } from '@/lib/ocean';

interface BasinLocationPickerProps {
  latitude: number;
  longitude: number;
  onSelectCoordinates: (lat: number, lon: number) => void;
  height?: string;
  transectMode?: boolean;
  transectPoints?: Array<{ latitude: number; longitude: number }>;
  onAddTransectPoint?: (lat: number, lon: number) => void;
  onClearTransect?: () => void;
  onToggleTransectMode?: () => void;
}

export const BasinLocationPicker: React.FC<BasinLocationPickerProps> = ({
  latitude,
  longitude,
  onSelectCoordinates,
  height = '380px',
  transectMode = false,
  transectPoints = [],
  onAddTransectPoint,
  onClearTransect,
  onToggleTransectMode,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.CircleMarker | null>(null);
  const pulseMarkerRef = useRef<L.CircleMarker | null>(null);

  // Transect layer group
  const transectLayerRef = useRef<L.LayerGroup | null>(null);

  const onSelectRef = useRef(onSelectCoordinates);
  const onAddTransectPointRef = useRef(onAddTransectPoint);
  const transectModeRef = useRef(transectMode);
  const initialCoordsRef = useRef({ latitude, longitude });

  useEffect(() => {
    onSelectRef.current = onSelectCoordinates;
    onAddTransectPointRef.current = onAddTransectPoint;
    transectModeRef.current = transectMode;
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
        if (transectModeRef.current && onAddTransectPointRef.current) {
          onAddTransectPointRef.current(preset.latitude, preset.longitude);
        } else {
          onSelectRef.current(preset.latitude, preset.longitude);
        }
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

    // Transect Layer Group
    const transectGroup = L.layerGroup().addTo(map);
    transectLayerRef.current = transectGroup;

    // Click on basin to clamp and snap to 0.25° grid
    map.on('click', (e: L.LeafletMouseEvent) => {
      const { latitude: snappedLat, longitude: snappedLon } = clampAndSnapCoordinates(
        e.latlng.lat,
        e.latlng.lng
      );
      if (transectModeRef.current && onAddTransectPointRef.current) {
        onAddTransectPointRef.current(snappedLat, snappedLon);
      } else {
        onSelectRef.current(snappedLat, snappedLon);
      }
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

  // Update Transect Line & Endpoint Markers on Map
  useEffect(() => {
    const group = transectLayerRef.current;
    if (!group) return;

    group.clearLayers();

    if (transectPoints.length === 0) return;

    // Draw waypoints
    transectPoints.forEach((pt, idx) => {
      const isStart = idx === 0;
      const isEnd = idx === transectPoints.length - 1 && transectPoints.length > 1;
      const markerColor = isStart ? '#059669' : isEnd ? '#4338ca' : '#f59e0b';
      const label = isStart ? 'A (Start)' : isEnd ? 'B (End)' : `P${idx + 1}`;

      const ptMarker = L.circleMarker([pt.latitude, pt.longitude], {
        radius: 6.5,
        color: '#ffffff',
        fillColor: markerColor,
        fillOpacity: 0.95,
        weight: 2,
      }).addTo(group);

      ptMarker.bindTooltip(
        `<div style="font-family: var(--font-sans, system-ui); font-size: 11px; padding: 2px;">
           <b style="color: ${markerColor};">${label}</b><br/>
           <span style="color: #64748d; font-family: var(--font-mono, monospace);">${pt.latitude.toFixed(2)}°N, ${pt.longitude.toFixed(2)}°E</span>
         </div>`,
        { permanent: true, direction: 'top', className: 'ocean-map-tooltip' }
      );
    });

    // Draw connecting polyline
    if (transectPoints.length >= 2) {
      const latlngs: L.LatLngExpression[] = transectPoints.map((p) => [p.latitude, p.longitude]);
      L.polyline(latlngs, {
        color: '#4338ca',
        weight: 3.0,
        dashArray: '6, 6',
        opacity: 0.85,
      }).addTo(group);
    }
  }, [transectPoints]);

  const handleResetView = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([16.5, 75.0], 4);
    }
  };

  return (
    <div
      className="relative isolate w-full h-full overflow-hidden bg-white"
      style={{ height }}
      role="region"
      aria-label="North Indian Ocean Basin Geographic Selector"
    >
      {/* Top Left Controls: Coordinates / Transect Status */}
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
        {!transectMode ? (
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-white/95 backdrop-blur-md border border-[#e3e8ee] text-xs font-mono text-[#0d253d] shadow-xs">
            <Crosshair className="size-3 text-[#533afd]" />
            <span className="tabular-nums font-medium">
              {latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[#4338ca] text-white text-xs font-mono shadow-xs">
            <Route className="size-3 text-white" />
            <span>
              {transectPoints.length === 0
                ? 'Click for Point A'
                : transectPoints.length === 1
                ? 'Click for Point B'
                : `Transect Active (${transectPoints.length} pts)`}
            </span>
            {transectPoints.length > 0 && onClearTransect && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onClearTransect();
                }}
                className="ml-1 p-0.5 rounded hover:bg-white/20 transition-colors"
                title="Clear transect"
              >
                <X className="size-3" />
              </button>
            )}
          </div>
        )}

        {/* Mode Toggle Button */}
        {onToggleTransectMode && (
          <button
            type="button"
            onClick={onToggleTransectMode}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border shadow-xs transition-all cursor-pointer ${
              transectMode
                ? 'bg-white text-[#4338ca] border-[#4338ca]'
                : 'bg-white/95 text-[#64748d] hover:text-[#0d253d] border-[#e3e8ee]'
            }`}
            title={transectMode ? 'Switch to Point Station Mode' : 'Switch to Vertical Transect Mode'}
          >
            {transectMode ? (
              <>
                <MapPin className="size-3 text-[#4338ca]" />
                <span>Station Mode</span>
              </>
            ) : (
              <>
                <Route className="size-3 text-[#4338ca]" />
                <span>Transect Mode</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Reset Map View Button */}
      <button
        type="button"
        onClick={handleResetView}
        title="Reset map view to North Indian Ocean domain"
        className="absolute top-3 right-12 z-20 p-1.5 rounded-lg bg-white/95 hover:bg-[#f8fafc] border border-[#e3e8ee] text-[#64748d] hover:text-[#0d253d] transition-all cursor-pointer shadow-xs active:scale-95"
        aria-label="Reset basin view"
      >
        <RotateCcw className="size-3.5" />
      </button>

      {/* Leaflet Map Canvas */}
      <div className="w-full h-full" ref={mapContainerRef} />

      {/* Domain Footnote */}
      <div className="absolute bottom-3 left-3 z-20 text-[11px] font-mono text-[#64748d] bg-white/95 backdrop-blur-md px-2.5 py-1 rounded-lg border border-[#e3e8ee] pointer-events-none shadow-xs">
        0.25° grid · 5°–30°N, 45°–105°E
      </div>
    </div>
  );
};
