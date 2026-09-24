import React, { useRef, useEffect, useState } from 'react';
import { Loader2, Route, RotateCcw, Compass, AlertTriangle } from 'lucide-react';
import type { TransectResponse, TransectStation } from '@/types/api';

interface VerticalTransectPlotProps {
  transect: TransectResponse | null;
  loading: boolean;
  onSelectPreset?: (points: Array<{ latitude: number; longitude: number }>) => void;
  onClearTransect?: () => void;
  height?: string | number;
}

// 15 Authoritative Standard Depths
const STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

// Oceanographic Turbo Colormap generator (temperatures: 4°C to 30°C)
function getTurboColor(t: number): [number, number, number] {
  const norm = Math.max(0, Math.min(1, (t - 4.0) / 26.0));
  // 5-stop Turbo/Ocean gradient
  const r = Math.sin(norm * Math.PI * 0.9 - 0.2);
  const g = Math.sin(norm * Math.PI * 0.9 + 0.3);
  const b = Math.cos(norm * Math.PI * 0.9 - 0.1);
  return [
    Math.round(Math.max(0, Math.min(1, 0.15 + 0.85 * r * r)) * 255),
    Math.round(Math.max(0, Math.min(1, 0.05 + 0.95 * g * g)) * 255),
    Math.round(Math.max(0, Math.min(1, 0.25 + 0.75 * b * b)) * 255),
  ];
}

// Turbo hex color for colorbar
function turboHex(norm: number): string {
  const [r, g, b] = getTurboColor(4.0 + norm * 26.0);
  return `rgb(${r}, ${g}, ${b})`;
}

export const VerticalTransectPlot: React.FC<VerticalTransectPlotProps> = ({
  transect,
  loading,
  onSelectPreset,
  onClearTransect,
  height = '100%',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    station: TransectStation;
    depth: number;
    temp: number | null;
  } | null>(null);

  const stations = transect?.stations || [];

  // Render 2D cross-section on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !transect || stations.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Margins for axes
    const padLeft = 60;
    const padRight = 30;
    const padTop = 24;
    const padBottom = 40;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = '#fafbfc';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    // Non-linear depth mapping for oceanographic clarity (more resolution in upper 200m)
    // Sqrt scale: y = sqrt(depth / 1000) * plotH
    const depthToY = (depth: number) => {
      return padTop + Math.sqrt(depth / 1000.0) * plotH;
    };
    const yToDepth = (y: number) => {
      const frac = Math.max(0, Math.min(1, (y - padTop) / plotH));
      return frac * frac * 1000.0;
    };

    const maxDist = transect.total_distance_km || 1;
    const distToX = (dist: number) => {
      return padLeft + (dist / maxDist) * plotW;
    };

    // 1. Draw 2D thermal field
    const imgData = ctx.createImageData(plotW, plotH);
    const data = imgData.data;

    const numStations = stations.length;

    for (let py = 0; py < plotH; py++) {
      const depth = yToDepth(padTop + py);
      // Find depth indices in STANDARD_DEPTHS
      let dIdx = 0;
      while (dIdx < STANDARD_DEPTHS.length - 1 && depth > STANDARD_DEPTHS[dIdx + 1]) {
        dIdx++;
      }
      const d1 = STANDARD_DEPTHS[dIdx];
      const d2 = STANDARD_DEPTHS[dIdx + 1] || d1;
      const dFrac = d2 === d1 ? 0 : (depth - d1) / (d2 - d1);

      for (let px = 0; px < plotW; px++) {
        const dist = (px / plotW) * maxDist;

        // Find station interval
        let sIdx = 0;
        while (sIdx < numStations - 1 && dist > stations[sIdx + 1].distance_km) {
          sIdx++;
        }
        const s1 = stations[sIdx];
        const s2 = stations[sIdx + 1] || s1;
        const sDist1 = s1.distance_km;
        const sDist2 = s2.distance_km;
        const sFrac = sDist2 === sDist1 ? 0 : (dist - sDist1) / (sDist2 - sDist1);

        const pIdx = (py * plotW + px) * 4;

        if (!s1.is_valid_ocean || !s2.is_valid_ocean || !s1.temperature_c || !s2.temperature_c) {
          // Land / unobserved: slate pattern
          const isHatch = (px + py) % 8 < 2;
          data[pIdx] = isHatch ? 203 : 226;
          data[pIdx + 1] = isHatch ? 213 : 232;
          data[pIdx + 2] = isHatch ? 225 : 240;
          data[pIdx + 3] = 255;
          continue;
        }

        // Interpolate temperature in depth and distance
        const t1_d1 = s1.temperature_c[dIdx] ?? 20;
        const t1_d2 = s1.temperature_c[dIdx + 1] ?? t1_d1;
        const t1 = t1_d1 + dFrac * (t1_d2 - t1_d1);

        const t2_d1 = s2.temperature_c[dIdx] ?? 20;
        const t2_d2 = s2.temperature_c[dIdx + 1] ?? t2_d1;
        const t2 = t2_d1 + dFrac * (t2_d2 - t2_d1);

        const tInterp = t1 + sFrac * (t2 - t1);

        const [r, g, b] = getTurboColor(tInterp);
        data[pIdx] = r;
        data[pIdx + 1] = g;
        data[pIdx + 2] = b;
        data[pIdx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, padLeft, padTop);

    // 2. Depth grid lines & labels
    ctx.font = '10px var(--font-mono, monospace)';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const labeledDepths = [0, 50, 100, 200, 500, 1000];
    labeledDepths.forEach((d) => {
      const y = depthToY(d);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#64748d';
      ctx.fillText(`${d}m`, padLeft - 6, y);
    });
    ctx.setLineDash([]);

    // 3. Distance grid lines & labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const numDistTicks = 5;
    for (let i = 0; i <= numDistTicks; i++) {
      const dist = (i / numDistTicks) * maxDist;
      const x = distToX(dist);

      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      ctx.fillStyle = '#475569';
      ctx.fillText(`${Math.round(dist)}km`, x, padTop + plotH + 6);
    }

    // 4. Draw D26 Isotherm Line (Green dashed)
    ctx.strokeStyle = '#059669';
    ctx.lineWidth = 2.0;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    let d26Started = false;
    stations.forEach((s) => {
      if (s.is_valid_ocean && s.d26_depth_m && s.d26_depth_m > 0) {
        const x = distToX(s.distance_km);
        const y = depthToY(s.d26_depth_m);
        if (!d26Started) {
          ctx.moveTo(x, y);
          d26Started = true;
        } else {
          ctx.lineTo(x, y);
        }
      } else {
        d26Started = false;
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // 5. Draw MLD Line (Orange dotted)
    ctx.strokeStyle = '#ea580c';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    let mldStarted = false;
    stations.forEach((s) => {
      if (s.is_valid_ocean && s.mixed_layer_depth_m && s.mixed_layer_depth_m > 0) {
        const x = distToX(s.distance_km);
        const y = depthToY(s.mixed_layer_depth_m);
        if (!mldStarted) {
          ctx.moveTo(x, y);
          mldStarted = true;
        } else {
          ctx.lineTo(x, y);
        }
      } else {
        mldStarted = false;
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // 6. Draw station dots along surface
    stations.forEach((s) => {
      const x = distToX(s.distance_km);
      ctx.fillStyle = s.is_valid_ocean ? '#4338ca' : '#94a3b8';
      ctx.beginPath();
      ctx.arc(x, padTop, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // 7. Outer border
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);
  }, [transect, stations]);

  // Handle canvas mouse move for interactive inspection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !transect || stations.length < 2) return;

    const rect = canvas.getBoundingClientRect();
    const xRatio = canvas.width / rect.width;
    const yRatio = canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * xRatio;
    const mouseY = (e.clientY - rect.top) * yRatio;

    const padLeft = 60;
    const padRight = 30;
    const padTop = 24;
    const padBottom = 40;

    const plotW = canvas.width - padLeft - padRight;
    const plotH = canvas.height - padTop - padBottom;

    if (
      mouseX < padLeft ||
      mouseX > padLeft + plotW ||
      mouseY < padTop ||
      mouseY > padTop + plotH
    ) {
      setHoverInfo(null);
      return;
    }

    const maxDist = transect.total_distance_km || 1;
    const dist = ((mouseX - padLeft) / plotW) * maxDist;

    // Find nearest station
    let nearest = stations[0];
    let minD = Math.abs(stations[0].distance_km - dist);
    for (let i = 1; i < stations.length; i++) {
      const d = Math.abs(stations[i].distance_km - dist);
      if (d < minD) {
        minD = d;
        nearest = stations[i];
      }
    }

    // Depth from y
    const yFrac = Math.max(0, Math.min(1, (mouseY - padTop) / plotH));
    const depth = Math.round(yFrac * yFrac * 1000.0);

    // Temperature at depth
    let temp: number | null = null;
    if (nearest.is_valid_ocean && nearest.temperature_c) {
      let dIdx = 0;
      while (dIdx < STANDARD_DEPTHS.length - 1 && depth > STANDARD_DEPTHS[dIdx + 1]) {
        dIdx++;
      }
      const d1 = STANDARD_DEPTHS[dIdx];
      const d2 = STANDARD_DEPTHS[dIdx + 1] || d1;
      const frac = d2 === d1 ? 0 : (depth - d1) / (d2 - d1);
      const t1 = nearest.temperature_c[dIdx];
      const t2 = nearest.temperature_c[dIdx + 1] || t1;
      temp = Math.round((t1 + frac * (t2 - t1)) * 100) / 100;
    }

    setHoverInfo({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      station: nearest,
      depth,
      temp,
    });
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  return (
    <div
      ref={containerRef}
      className="w-full h-full flex flex-col bg-white overflow-hidden select-none relative"
      style={{ minHeight: '340px', height }}
    >
      {/* Top Controls & Metadata Bar */}
      <div className="px-4 py-2 bg-[#f8fafc] border-b border-[#e2e8f0] flex flex-wrap items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-2">
          <Route className="size-4 text-[#4338ca]" />
          <span className="text-xs font-semibold text-[#0d253d] tracking-wide uppercase">
            2D Vertical Subsurface Transect
          </span>
          {transect && (
            <span className="text-xs font-mono text-[#64748d]">
              · {transect.total_distance_km} km ({stations.length} stations)
            </span>
          )}
        </div>

        {/* Preset Transect Selectors */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-[#64748d] hidden sm:inline">Presets:</span>
          {onSelectPreset && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() =>
                  onSelectPreset([
                    { latitude: 15.0, longitude: 58.0 },
                    { latitude: 15.0, longitude: 73.0 },
                  ])
                }
                className="px-2 py-0.5 rounded text-[11px] font-medium bg-white hover:bg-[#f1f5f9] text-[#4338ca] border border-[#cbd5e1] shadow-2xs transition-colors cursor-pointer"
                title="Arabian Sea Zonal Transect (15°N)"
              >
                Arabian Sea (15°N)
              </button>

              <button
                type="button"
                onClick={() =>
                  onSelectPreset([
                    { latitude: 8.0, longitude: 88.0 },
                    { latitude: 20.0, longitude: 88.0 },
                  ])
                }
                className="px-2 py-0.5 rounded text-[11px] font-medium bg-white hover:bg-[#f1f5f9] text-[#4338ca] border border-[#cbd5e1] shadow-2xs transition-colors cursor-pointer"
                title="Bay of Bengal Meridional Transect (88°E)"
              >
                Bay of Bengal (88°E)
              </button>

              <button
                type="button"
                onClick={() =>
                  onSelectPreset([
                    { latitude: 5.0, longitude: 60.0 },
                    { latitude: 5.0, longitude: 92.0 },
                  ])
                }
                className="px-2 py-0.5 rounded text-[11px] font-medium bg-white hover:bg-[#f1f5f9] text-[#4338ca] border border-[#cbd5e1] shadow-2xs transition-colors cursor-pointer"
                title="Equatorial NIO Transect (5°N)"
              >
                Equatorial (5°N)
              </button>
            </div>
          )}

          {onClearTransect && (
            <button
              type="button"
              onClick={onClearTransect}
              className="p-1 rounded text-[#64748d] hover:text-[#0d253d] hover:bg-[#e2e8f0] transition-colors cursor-pointer"
              title="Reset transect"
            >
              <RotateCcw className="size-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Plot Stage */}
      <div className="flex-1 min-h-0 relative w-full overflow-hidden p-3 flex flex-col">
        {loading ? (
          <div className="w-full h-full flex flex-col items-center justify-center gap-2 text-[#64748d]">
            <Loader2 className="size-6 animate-spin text-[#4338ca]" />
            <span className="text-xs font-mono">Reconstructing 2D Vertical Transect...</span>
          </div>
        ) : !transect || stations.length < 2 ? (
          <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center text-[#64748d] bg-[#f8fafc] rounded-lg border border-dashed border-[#cbd5e1]">
            <Compass className="size-8 text-[#94a3b8] mb-2" />
            <h4 className="text-sm font-semibold text-[#0d253d] mb-1">
              No Transect Defined
            </h4>
            <p className="text-xs max-w-md text-[#64748d] mb-4">
              Click two points on the North Indian Ocean map in <b>Transect Mode</b> to define a vertical cross-section, or pick one of the oceanographic presets above.
            </p>
            {onSelectPreset && (
              <button
                type="button"
                onClick={() =>
                  onSelectPreset([
                    { latitude: 15.0, longitude: 58.0 },
                    { latitude: 15.0, longitude: 73.0 },
                  ])
                }
                className="px-3 py-1.5 text-xs font-medium bg-[#4338ca] text-white rounded-lg shadow-xs hover:bg-[#3730a3] transition-colors cursor-pointer"
              >
                Load Arabian Sea Zonal Section
              </button>
            )}
          </div>
        ) : (
          <div className="flex-1 w-full h-full relative">
            <canvas
              ref={canvasRef}
              width={820}
              height={340}
              className="w-full h-full object-contain cursor-crosshair"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            />

            {/* Hover Tooltip Box */}
            {hoverInfo && (
              <div
                className="absolute pointer-events-none z-20 px-3 py-2 rounded-lg bg-[#0d253d]/95 backdrop-blur-sm text-white shadow-xl text-xs font-mono space-y-1 border border-white/20"
                style={{
                  left: `${Math.min(hoverInfo.x + 15, 620)}px`,
                  top: `${Math.max(10, hoverInfo.y - 40)}px`,
                }}
              >
                <div className="flex items-center justify-between gap-4 border-b border-white/20 pb-1">
                  <span className="font-semibold text-white">
                    Station #{hoverInfo.station.index + 1}
                  </span>
                  <span className="text-[#93c5fd]">
                    {hoverInfo.station.distance_km} km
                  </span>
                </div>
                <div className="text-[#cbd5e1] text-[11px]">
                  {hoverInfo.station.latitude.toFixed(2)}°N, {hoverInfo.station.longitude.toFixed(2)}°E
                </div>

                {hoverInfo.station.is_valid_ocean ? (
                  <>
                    <div className="flex items-center justify-between gap-3 text-white">
                      <span>Depth: {hoverInfo.depth}m</span>
                      <strong className="text-amber-300">
                        {hoverInfo.temp !== null ? `${hoverInfo.temp.toFixed(2)}°C` : '—'}
                      </strong>
                    </div>
                    {hoverInfo.station.d26_depth_m !== null && hoverInfo.station.d26_depth_m !== undefined && (
                      <div className="text-[11px] text-[#34d399]">
                        D26 Isotherm: {hoverInfo.station.d26_depth_m.toFixed(1)}m
                      </div>
                    )}
                    {hoverInfo.station.mixed_layer_depth_m !== null && hoverInfo.station.mixed_layer_depth_m !== undefined && (
                      <div className="text-[11px] text-[#fb923c]">
                        Mixed Layer (MLD): {hoverInfo.station.mixed_layer_depth_m.toFixed(1)}m
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-rose-300 font-medium flex items-center gap-1 mt-1">
                    <AlertTriangle className="size-3" /> Land / Unobserved Cell
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Legend & Thermal Scale Footer */}
      <div className="px-4 py-2 bg-[#fafbfc] border-t border-[#e2e8f0] flex flex-wrap items-center justify-between gap-3 text-xs shrink-0">
        {/* Isobar Legend */}
        <div className="flex items-center gap-4 text-[#475569]">
          <div className="flex items-center gap-1.5 font-mono text-[11px]">
            <span className="inline-block w-4 h-0.5 border-t-2 border-dashed border-[#059669]" />
            <span>D26 Isotherm (26°C)</span>
          </div>

          <div className="flex items-center gap-1.5 font-mono text-[11px]">
            <span className="inline-block w-4 h-0.5 border-t-2 border-dotted border-[#ea580c]" />
            <span>Mixed Layer (MLD)</span>
          </div>

          <div className="flex items-center gap-1.5 font-mono text-[11px]">
            <span className="inline-block size-2.5 bg-[#cbd5e1] border border-[#94a3b8]" />
            <span>Land / Mask</span>
          </div>
        </div>

        {/* Continuous Colormap Scale */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-[#64748d]">4°C</span>
          <div
            className="w-32 h-2.5 rounded shadow-2xs border border-[#cbd5e1]"
            style={{
              background: `linear-gradient(to right, ${turboHex(0)}, ${turboHex(0.25)}, ${turboHex(0.5)}, ${turboHex(0.75)}, ${turboHex(1)})`,
            }}
          />
          <span className="text-[11px] font-mono text-[#64748d]">30°C</span>
        </div>
      </div>
    </div>
  );
};
