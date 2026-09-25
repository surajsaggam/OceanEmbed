import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Loader2, Info, Compass, Layers, AlertCircle } from 'lucide-react';
import { fetchDeparture } from '@/services/api';
import type { DepartureResponse, DepthDepartureMetrics, StationDepartureProfile } from '@/types/api';

interface ReconstructionDeparturePanelProps {
  date: string;
  latitude: number;
  longitude: number;
  isMock: boolean;
  onSelectDate?: (date: string) => void;
}

const STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

// Scientifically appropriate diverging color generator (-2.5°C to +2.5°C)
// Negative (blue) = OceanIQ colder than GLORYS
// Zero (white) = Agreement
// Positive (red) = OceanIQ warmer than GLORYS
function getDivergingColor(dep: number, maxAbs = 2.5): [number, number, number] {
  const norm = Math.max(-1.0, Math.min(1.0, dep / maxAbs));
  if (norm < 0) {
    // Negative: White (250, 250, 252) -> Deep Royal Blue (29, 78, 216)
    const t = -norm;
    const r = Math.round(250 - t * (250 - 29));
    const g = Math.round(250 - t * (250 - 78));
    const b = Math.round(252 - t * (252 - 216));
    return [r, g, b];
  } else {
    // Positive: White (250, 250, 252) -> Deep Crimson Red (220, 38, 38)
    const t = norm;
    const r = Math.round(250 - t * (250 - 220));
    const g = Math.round(250 - t * (250 - 38));
    const b = Math.round(252 - t * (252 - 38));
    return [r, g, b];
  }
}

export const ReconstructionDeparturePanel: React.FC<ReconstructionDeparturePanelProps> = ({
  date,
  latitude,
  longitude,
  isMock,
  onSelectDate,
}) => {
  const [selectedDepth, setSelectedDepth] = useState<number>(100);
  const [departureData, setDepartureData] = useState<DepartureResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{
    lat: number;
    lon: number;
    dep: number | null;
    x: number;
    y: number;
  } | null>(null);

  // Fetch departure data
  const loadDeparture = useCallback(
    async (depth: number, targetDate: string) => {
      setLoading(true);
      setErrorMsg(null);
      try {
        const res = await fetchDeparture({
          date: targetDate,
          depth_m: depth,
          latitude,
          longitude,
        });
        setDepartureData(res);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Failed to calculate departure field.';
        setErrorMsg(msg);
        setDepartureData(null);
      } finally {
        setLoading(false);
      }
    },
    [latitude, longitude]
  );

  useEffect(() => {
    loadDeparture(selectedDepth, date);
  }, [selectedDepth, date, loadDeparture]);

  // Render 2D Departure Basin Grid on Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !departureData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { grid_lat, grid_lon, grid_departure } = departureData;
    const nLat = grid_lat.length;
    const nLon = grid_lon.length;
    if (nLat === 0 || nLon === 0) return;

    const width = canvas.width;
    const height = canvas.height;

    // Background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, width, height);

    // Padding for geographic axes
    const padLeft = 45;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 30;

    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const cellW = plotWidth / nLon;
    const cellH = plotHeight / nLat;

    // Draw cells (lats go from 5N at bottom to 30N at top)
    for (let i = 0; i < nLat; i++) {
      // In grid_lat, index 0 is 5.0N, index nLat-1 is 30.0N
      // In canvas, y=0 is top (30N), y=plotHeight is bottom (5N)
      const y = padTop + (nLat - 1 - i) * cellH;

      for (let j = 0; j < nLon; j++) {
        const x = padLeft + j * cellW;
        const val = grid_departure[i]?.[j];

        if (val === null || val === undefined) {
          // Land / unmasked cell
          ctx.fillStyle = '#e2e8f0';
        } else {
          const [r, g, b] = getDivergingColor(val, 2.5);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        }
        ctx.fillRect(x, y, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5);
      }
    }

    // Grid border
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotWidth, plotHeight);

    // Geographic Axis Labels
    ctx.fillStyle = '#64748d';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    // Latitude labels
    [5, 10, 15, 20, 25, 30].forEach((lat) => {
      const frac = (lat - 5) / 25;
      const y = padTop + (1 - frac) * plotHeight;
      ctx.fillText(`${lat}°N`, padLeft - 6, y);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotWidth, y);
      ctx.stroke();
    });

    // Longitude labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    [50, 60, 70, 80, 90, 100].forEach((lon) => {
      const frac = (lon - 45) / 60;
      const x = padLeft + frac * plotWidth;
      ctx.fillText(`${lon}°E`, x, padTop + plotHeight + 6);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotHeight);
      ctx.stroke();
    });

    // Draw active station pin
    if (latitude >= 5 && latitude <= 30 && longitude >= 45 && longitude <= 105) {
      const stationX = padLeft + ((longitude - 45) / 60) * plotWidth;
      const stationY = padTop + (1 - (latitude - 5) / 25) * plotHeight;

      ctx.strokeStyle = '#0d253d';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(stationX, stationY, 6, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(stationX, stationY, 4, 0, Math.PI * 2);
      ctx.fill();

      // Crosshairs
      ctx.strokeStyle = 'rgba(13, 37, 61, 0.6)';
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(stationX, padTop);
      ctx.lineTo(stationX, padTop + plotHeight);
      ctx.moveTo(padLeft, stationY);
      ctx.lineTo(padLeft + plotWidth, stationY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [departureData, latitude, longitude]);

  // Handle canvas mouse move for interactive inspection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !departureData) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padLeft = 45;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 30;

    const plotWidth = canvas.width - padLeft - padRight;
    const plotHeight = canvas.height - padTop - padBottom;

    if (x >= padLeft && x <= padLeft + plotWidth && y >= padTop && y <= padTop + plotHeight) {
      const fracX = (x - padLeft) / plotWidth;
      const fracY = 1.0 - (y - padTop) / plotHeight;

      const lon = 45.0 + fracX * 60.0;
      const lat = 5.0 + fracY * 25.0;

      // Find nearest grid indices
      const { grid_lat, grid_lon, grid_departure } = departureData;
      const latIdx = Math.max(0, Math.min(grid_lat.length - 1, Math.round(fracY * (grid_lat.length - 1))));
      const lonIdx = Math.max(0, Math.min(grid_lon.length - 1, Math.round(fracX * (grid_lon.length - 1))));
      const val = grid_departure[latIdx]?.[lonIdx] ?? null;

      setHoveredCell({
        lat: Math.round(lat * 10) / 10,
        lon: Math.round(lon * 10) / 10,
        dep: val,
        x,
        y,
      });
    } else {
      setHoveredCell(null);
    }
  };

  const currentMetrics: DepthDepartureMetrics | undefined = departureData?.depth_metrics;
  const stationProf: StationDepartureProfile | null | undefined = departureData?.station_profile;

  return (
    <div className="space-y-4">
      {/* Scientific Orientation & Lineage Banner */}
      <div className="bg-[#f8fafc] border border-[#e2e8f0] rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
              OceanIQ Reconstruction Departure
            </span>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-[#e2e8f0] text-[#334155] font-medium">
              vs GLORYS12V1 Reanalysis Reference
            </span>
            {departureData?.is_mock && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-[#fef3c7] text-[#92400e] font-semibold">
                DEMO / SYNTHETIC
              </span>
            )}
          </div>
          <p className="text-xs text-[#64748d] font-mono leading-relaxed">
            Departure = OceanIQ reconstruction − GLORYS12V1 reference · Held-out evaluation date: {departureData?.date || '2019-01-01'}
          </p>
        </div>

        {/* Date status badge / switcher */}
        {date !== '2019-01-01' && !isMock && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#b45309] font-mono flex items-center gap-1">
              <AlertCircle className="w-3.5 h-3.5" />
              Viewing {date} (Ref only at 2019-01-01)
            </span>
            {onSelectDate && (
              <button
                type="button"
                onClick={() => onSelectDate('2019-01-01')}
                className="text-xs font-mono font-medium px-2.5 py-1 rounded bg-[#0284c7] text-white hover:bg-[#0369a1] transition-all cursor-pointer shadow-xs"
              >
                Inspect 2019-01-01
              </button>
            )}
          </div>
        )}
      </div>

      {/* Depth Level Selector */}
      <div className="bg-white border border-[#e2e8f0] rounded-xl p-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
        <div className="flex items-center gap-2 shrink-0">
          <Layers className="w-4 h-4 text-[#0284c7]" />
          <span className="text-xs font-semibold text-[#0d253d] uppercase tracking-wider">
            Analysis Depth Level:
          </span>
          <span className="text-xs font-mono font-bold text-[#0284c7]">
            {selectedDepth} m
          </span>
        </div>

        {/* 15 Standard Depths Horizontal Selector */}
        <div className="flex flex-wrap items-center gap-1">
          {STANDARD_DEPTHS.map((depth) => (
            <button
              key={depth}
              type="button"
              onClick={() => setSelectedDepth(depth)}
              disabled={loading}
              className={`px-2 py-1 text-[11px] font-mono font-medium rounded transition-all cursor-pointer ${
                selectedDepth === depth
                  ? 'bg-[#0d253d] text-white shadow-xs'
                  : 'bg-[#f1f5f9] text-[#64748d] hover:bg-[#e2e8f0] hover:text-[#0d253d]'
              }`}
            >
              {depth}m
            </button>
          ))}
        </div>
      </div>

      {/* Main Body: Diverging Spatial Map & Statistical Scorecard */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: 2D Spatial Departure Field Canvas (lg:col-span-7) */}
        <div className="lg:col-span-7 bg-white border border-[#e2e8f0] rounded-xl p-4 shadow-xs flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-[#f1f5f9]">
            <div className="flex items-center gap-2">
              <Compass className="w-4 h-4 text-[#0d253d]" />
              <span className="text-xs font-semibold text-[#0d253d] uppercase tracking-wider">
                Spatial Departure Field ({selectedDepth}m)
              </span>
            </div>
            <span className="text-[11px] font-mono text-[#64748d]">
              Grid: 5°N–30°N, 45°E–105°E
            </span>
          </div>

          {/* Canvas Rendering Area */}
          <div className="relative mt-3 flex-1 min-h-[300px] flex items-center justify-center bg-[#f8fafc] rounded-lg overflow-hidden border border-[#e2e8f0]">
            {loading && (
              <div className="absolute inset-0 z-10 bg-white/70 backdrop-blur-xs flex items-center justify-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin text-[#0284c7]" />
                <span className="text-xs font-mono text-[#0d253d]">Computing departure field...</span>
              </div>
            )}

            {errorMsg ? (
              <div className="p-6 text-center text-xs font-mono text-[#b91c1c] max-w-md">
                <AlertCircle className="w-6 h-6 mx-auto mb-2 text-[#ef4444]" />
                {errorMsg}
              </div>
            ) : (
              <canvas
                ref={canvasRef}
                width={540}
                height={300}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoveredCell(null)}
                className="w-full h-auto max-h-[340px] cursor-crosshair block"
              />
            )}

            {/* Hover Tooltip */}
            {hoveredCell && (
              <div
                className="absolute z-20 pointer-events-none bg-[#0d253d]/90 text-white px-2.5 py-1.5 rounded text-[11px] font-mono shadow-lg border border-[#334155]"
                style={{
                  left: Math.min(hoveredCell.x + 12, 380),
                  top: Math.max(hoveredCell.y - 40, 10),
                }}
              >
                <div>Coord: {hoveredCell.lat.toFixed(1)}°N, {hoveredCell.lon.toFixed(1)}°E</div>
                <div>
                  Departure:{' '}
                  {hoveredCell.dep !== null ? (
                    <span
                      className={
                        hoveredCell.dep > 0
                          ? 'text-[#fca5a5] font-bold'
                          : hoveredCell.dep < 0
                          ? 'text-[#93c5fd] font-bold'
                          : 'text-[#86efac] font-bold'
                      }
                    >
                      {hoveredCell.dep > 0 ? '+' : ''}
                      {hoveredCell.dep.toFixed(2)} °C
                    </span>
                  ) : (
                    <span className="text-[#94a3b8]">Land / Masked</span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Diverging Colorbar Legend */}
          <div className="mt-3 pt-3 border-t border-[#f1f5f9] flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-[11px] font-mono text-[#64748d]">
              <span className="text-[#1d4ed8] font-medium">-2.5 °C (OceanIQ Colder)</span>
              <span className="text-[#334155] font-semibold">0.0 °C (Agreement)</span>
              <span className="text-[#dc2626] font-medium">+2.5 °C (OceanIQ Warmer)</span>
            </div>
            <div
              className="h-2.5 w-full rounded-full border border-[#cbd5e1]"
              style={{
                background:
                  'linear-gradient(to right, rgb(29, 78, 216), rgb(147, 197, 253), rgb(250, 250, 252), rgb(252, 165, 165), rgb(220, 38, 38))',
              }}
            />
          </div>
        </div>

        {/* Right: Depth Statistics Scorecard & Station Profile (lg:col-span-5) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Depth Metric Scorecards */}
          <div className="bg-white border border-[#e2e8f0] rounded-xl p-4 shadow-xs">
            <div className="flex items-center justify-between pb-3 border-b border-[#f1f5f9]">
              <span className="text-xs font-semibold text-[#0d253d] uppercase tracking-wider">
                Evaluation Metrics ({selectedDepth}m)
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#f1f5f9] text-[#64748d]">
                {currentMetrics ? `${currentMetrics.valid_cells.toLocaleString()} Ocean Cells` : 'Valid Cells'}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 mt-3">
              <div className="p-3 bg-[#f8fafc] border border-[#e2e8f0] rounded-lg text-center">
                <span className="text-[11px] font-mono text-[#64748d] block">RMSE</span>
                <span className="text-lg font-mono font-bold text-[#0d253d]">
                  {currentMetrics ? `${currentMetrics.rmse.toFixed(3)}` : '—'}
                </span>
                <span className="text-[10px] font-mono text-[#94a3b8]">°C</span>
              </div>

              <div className="p-3 bg-[#f8fafc] border border-[#e2e8f0] rounded-lg text-center">
                <span className="text-[11px] font-mono text-[#64748d] block">MAE</span>
                <span className="text-lg font-mono font-bold text-[#0d253d]">
                  {currentMetrics ? `${currentMetrics.mae.toFixed(3)}` : '—'}
                </span>
                <span className="text-[10px] font-mono text-[#94a3b8]">°C</span>
              </div>

              <div className="p-3 bg-[#f8fafc] border border-[#e2e8f0] rounded-lg text-center">
                <span className="text-[11px] font-mono text-[#64748d] block">Mean Bias</span>
                <span
                  className={`text-lg font-mono font-bold ${
                    (currentMetrics?.mean_bias ?? 0) > 0
                      ? 'text-[#dc2626]'
                      : (currentMetrics?.mean_bias ?? 0) < 0
                      ? 'text-[#2563eb]'
                      : 'text-[#0d253d]'
                  }`}
                >
                  {currentMetrics
                    ? `${currentMetrics.mean_bias >= 0 ? '+' : ''}${currentMetrics.mean_bias.toFixed(3)}`
                    : '—'}
                </span>
                <span className="text-[10px] font-mono text-[#94a3b8]">°C</span>
              </div>
            </div>

            {currentMetrics && (
              <div className="mt-3 pt-2 text-[11px] font-mono text-[#64748d] flex justify-between border-t border-[#f1f5f9]">
                <span>Min: {currentMetrics.min_departure_c.toFixed(2)} °C</span>
                <span>Max: +{currentMetrics.max_departure_c.toFixed(2)} °C</span>
              </div>
            )}
          </div>

          {/* Active Station Vertical Sounding Departure Schedule */}
          <div className="bg-white border border-[#e2e8f0] rounded-xl p-4 shadow-xs">
            <div className="flex items-center justify-between pb-2 border-b border-[#f1f5f9]">
              <span className="text-xs font-semibold text-[#0d253d] uppercase tracking-wider">
                Station Profile Comparison
              </span>
              <span className="text-[11px] font-mono text-[#64748d]">
                {latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E
              </span>
            </div>

            {stationProf ? (
              <div className="mt-2.5">
                <div className="text-[11px] font-mono text-[#64748d] mb-2 flex items-center justify-between">
                  <span>Column Mean Absolute Departure:</span>
                  <span className="font-bold text-[#0d253d]">
                    {stationProf.mean_absolute_departure_c.toFixed(3)} °C
                  </span>
                </div>

                {/* Compact Sounding Residual Table */}
                <div className="max-h-[190px] overflow-y-auto border border-[#e2e8f0] rounded-lg">
                  <table className="w-full text-left text-[11px] font-mono">
                    <thead className="bg-[#f8fafc] text-[#64748d] sticky top-0 border-b border-[#e2e8f0]">
                      <tr>
                        <th className="py-1 px-2.5">Depth</th>
                        <th className="py-1 px-2 text-right">OceanIQ</th>
                        <th className="py-1 px-2 text-right">GLORYS</th>
                        <th className="py-1 px-2.5 text-right">Departure</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#f1f5f9]">
                      {stationProf.depths_m.map((d, idx) => {
                        const rec = stationProf.reconstructed_c[idx];
                        const ref = stationProf.reference_c[idx];
                        const dep = stationProf.departure_c[idx];
                        const isSelected = d === selectedDepth;

                        return (
                          <tr
                            key={d}
                            className={`cursor-pointer transition-colors ${
                              isSelected ? 'bg-[#f0f9ff] font-semibold' : 'hover:bg-[#f8fafc]'
                            }`}
                            onClick={() => setSelectedDepth(d)}
                          >
                            <td className="py-1 px-2.5 text-[#0d253d]">
                              {d}m {isSelected && '•'}
                            </td>
                            <td className="py-1 px-2 text-right text-[#334155]">{rec.toFixed(2)}°C</td>
                            <td className="py-1 px-2 text-right text-[#64748d]">{ref.toFixed(2)}°C</td>
                            <td
                              className={`py-1 px-2.5 text-right font-medium ${
                                dep > 0 ? 'text-[#dc2626]' : dep < 0 ? 'text-[#2563eb]' : 'text-[#64748d]'
                              }`}
                            >
                              {dep >= 0 ? '+' : ''}
                              {dep.toFixed(2)}°C
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="py-8 text-center text-xs font-mono text-[#94a3b8]">
                Select an ocean station to view vertical sounding departure.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Scientific Footnote */}
      <div className="bg-[#f1f5f9]/70 border border-[#e2e8f0] rounded-lg p-3 text-[11px] font-mono text-[#64748d] flex items-start gap-2">
        <Info className="w-4 h-4 text-[#0284c7] shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-[#0d253d]">Scientific Integrity Note: </span>
          {departureData?.scientific_note ||
            'Departure = OceanIQ reconstruction − GLORYS12V1 reference. Evaluated on the held-out 2019-01-01 test date. This represents a model reconstruction departure from reanalysis, NOT a climatological anomaly or marine heatwave.'}
        </div>
      </div>
    </div>
  );
};
