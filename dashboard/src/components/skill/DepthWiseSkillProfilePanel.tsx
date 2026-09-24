import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Loader2, Info, Layers, TrendingDown, AlertCircle } from 'lucide-react';
import { fetchDeparture } from '@/services/api';
import type { DepthDepartureMetrics, DepartureResponse } from '@/types/api';

interface DepthWiseSkillProfilePanelProps {
  date: string;
  isMock: boolean;
  onSelectDate?: (date: string) => void;
}

const STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

export const DepthWiseSkillProfilePanel: React.FC<DepthWiseSkillProfilePanelProps> = ({
  date,
  isMock,
  onSelectDate,
}) => {
  const [metrics, setMetrics] = useState<DepthDepartureMetrics[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedDepth, setSelectedDepth] = useState<number | null>(100);
  const [hoveredDepth, setHoveredDepth] = useState<number | null>(null);
  const [hoverCoords, setHoverCoords] = useState<{ x: number; y: number } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load metrics from existing departure API
  useEffect(() => {
    let isCancelled = false;
    async function loadData() {
      setLoading(true);
      setErrorMsg(null);
      try {
        const res: DepartureResponse = await fetchDeparture({
          date,
          depth_m: 100,
        });
        if (!isCancelled) {
          if (res.all_depth_metrics && res.all_depth_metrics.length > 0) {
            setMetrics(res.all_depth_metrics);
          } else if (res.depth_metrics) {
            setMetrics([res.depth_metrics]);
          }
        }
      } catch (err: unknown) {
        if (!isCancelled) {
          const msg = err instanceof Error ? err.message : 'Failed to load depth-wise skill metrics.';
          setErrorMsg(msg);
        }
      } finally {
        if (!isCancelled) setLoading(false);
      }
    }
    loadData();
    return () => {
      isCancelled = true;
    };
  }, [date]);

  // Summary statistics
  const summary = useMemo(() => {
    if (metrics.length === 0) return null;
    const peakRmse = [...metrics].sort((a, b) => b.rmse - a.rmse)[0];
    const surface = metrics.find((m) => m.depth_m === 0) || metrics[0];
    const abyss = metrics.find((m) => m.depth_m === 1000) || metrics[metrics.length - 1];
    const meanBias = metrics.reduce((acc, m) => acc + m.mean_bias, 0) / metrics.length;
    return { peakRmse, surface, abyss, meanBias };
  }, [metrics]);

  const activeDepth = hoveredDepth ?? selectedDepth;

  // Canvas drawing for the primary error curve (Oceanographic vertical sounding)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // HiDPI / Retina Crisp Scaling
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = 540;
    const displayHeight = 350;

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;

    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, displayWidth, displayHeight);

    // Clean background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    const padLeft = 60;
    const padRight = 32;
    const padTop = 20;
    const padBottom = 38;

    const plotWidth = displayWidth - padLeft - padRight;
    const plotHeight = displayHeight - padTop - padBottom;

    const maxError = 1.4; // Max error in °C on axis

    // Oceanographic vertical sounding:
    // Vertical axis = Depth (0m at top -> 1000m at bottom)
    // Horizontal axis = Error in °C (0.0 -> maxError)
    const depthY = (d: number): number => {
      const idx = STANDARD_DEPTHS.indexOf(d);
      if (idx !== -1) {
        return padTop + (idx / (STANDARD_DEPTHS.length - 1)) * plotHeight;
      }
      return padTop + (Math.log10(Math.max(1, d) + 1) / Math.log10(1001)) * plotHeight;
    };

    const errorX = (err: number): number => {
      return padLeft + (Math.max(0, err) / maxError) * plotWidth;
    };

    // Subtle Thermocline wash (50m to 150m) - Restrained, clean, non-distracting
    const tcTop = depthY(50);
    const tcBottom = depthY(150);
    ctx.fillStyle = 'rgba(241, 245, 249, 0.65)';
    ctx.fillRect(padLeft, tcTop, plotWidth, tcBottom - tcTop);

    // Delicate dashed boundaries for thermocline
    ctx.strokeStyle = 'rgba(203, 213, 225, 0.7)';
    ctx.lineWidth = 0.75;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padLeft, tcTop);
    ctx.lineTo(padLeft + plotWidth, tcTop);
    ctx.moveTo(padLeft, tcBottom);
    ctx.lineTo(padLeft + plotWidth, tcBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Restrained thermocline annotation (positioned at top of zone to avoid crosshair collision)
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('Thermocline Layer (50–150m)', padLeft + plotWidth - 8, tcTop + 4);


    // Simplified, light gridlines for standard depths
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.7)';
    ctx.lineWidth = 0.75;
    STANDARD_DEPTHS.forEach((d) => {
      const y = depthY(d);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotWidth, y);
      ctx.stroke();

      // Axis ticks at left border
      ctx.strokeStyle = '#cbd5e1';
      ctx.beginPath();
      ctx.moveTo(padLeft - 3, y);
      ctx.lineTo(padLeft, y);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.7)';

      // Depth labels on Y-axis
      ctx.fillStyle = '#64748d';
      ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${d}m`, padLeft - 6, y);
    });

    // Vertical grid lines for error intervals (0.2°C steps)
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4].forEach((e) => {
      const x = errorX(e);
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.7)';
      ctx.lineWidth = 0.75;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotHeight);
      ctx.stroke();

      // Axis tick at bottom border
      ctx.strokeStyle = '#cbd5e1';
      ctx.beginPath();
      ctx.moveTo(x, padTop + plotHeight);
      ctx.lineTo(x, padTop + plotHeight + 3);
      ctx.stroke();

      ctx.fillStyle = '#64748d';
      ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${e.toFixed(1)}°`, x, padTop + plotHeight + 6);
    });

    // Axes labels
    ctx.fillStyle = '#334155';
    ctx.font = '10.5px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Reconstruction Error (°C)', padLeft + plotWidth / 2, padTop + plotHeight + 23);

    ctx.save();
    ctx.translate(14, padTop + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Depth (m) ↓', 0, 0);
    ctx.restore();

    // Secondary curve: MAE (Thinner, dashed, muted slate-indigo `#6366f1`)
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    metrics.forEach((m, i) => {
      const x = errorX(m.mae);
      const y = depthY(m.depth_m);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Primary dominant curve: RMSE (Authoritative deep ocean navy `#0284c7`, solid)
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    metrics.forEach((m, i) => {
      const x = errorX(m.rmse);
      const y = depthY(m.depth_m);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Data points & active selection
    metrics.forEach((m) => {
      const y = depthY(m.depth_m);
      const xR = errorX(m.rmse);
      const xM = errorX(m.mae);
      const isHighlight = m.depth_m === activeDepth;

      // MAE marker (small discrete square)
      ctx.fillStyle = '#6366f1';
      ctx.fillRect(
        xM - (isHighlight ? 3.5 : 2.5),
        y - (isHighlight ? 3.5 : 2.5),
        isHighlight ? 7 : 5,
        isHighlight ? 7 : 5
      );

      // RMSE marker (dominant circle)
      ctx.fillStyle = isHighlight ? '#0369a1' : '#0284c7';
      ctx.beginPath();
      ctx.arc(xR, y, isHighlight ? 5.0 : 3.2, 0, Math.PI * 2);
      ctx.fill();

      if (isHighlight) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Subtle crosshair guideline across hovered depth
        ctx.strokeStyle = 'rgba(2, 132, 199, 0.35)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(padLeft + plotWidth, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Outer framing border
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotWidth, plotHeight);

    ctx.restore();
  }, [metrics, activeDepth]);

  // Handle canvas mouse move for interactive depth hovering
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    const scaleY = 350 / (rect.height || 350);
    const y = clientY * scaleY;

    const padTop = 20;
    const padBottom = 38;
    const plotHeight = 350 - padTop - padBottom;

    if (y >= padTop && y <= padTop + plotHeight) {
      const frac = (y - padTop) / plotHeight;
      const idx = Math.max(
        0,
        Math.min(STANDARD_DEPTHS.length - 1, Math.round(frac * (STANDARD_DEPTHS.length - 1)))
      );
      setHoveredDepth(STANDARD_DEPTHS[idx]);
      setHoverCoords({ x: clientX, y: clientY });
    } else {
      setHoveredDepth(null);
      setHoverCoords(null);
    }
  };


  const handleCanvasMouseLeave = () => {
    setHoveredDepth(null);
    setHoverCoords(null);
  };

  const activeMetric = metrics.find((m) => m.depth_m === activeDepth) || metrics[0];


  return (
    <div className="space-y-4">
      {/* Scientific Lineage & Context Banner */}
      <div className="bg-[#f8fafc] border border-[#e2e8f0] rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
              Depth-wise Model Skill Profile
            </span>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-[#e2e8f0] text-[#334155] font-medium">
              OceanIQ vs GLORYS12V1 Reference
            </span>
            {isMock && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-[#fef3c7] text-[#92400e] font-semibold">
                DEMO / SYNTHETIC
              </span>
            )}
          </div>
          <p className="text-xs text-[#64748d] font-mono leading-relaxed">
            Reconstruction Error vs Depth · Evaluated across all valid ocean cells on held-out reference date: {date}
          </p>
        </div>

        {/* Date notice / switcher */}
        {date !== '2019-01-01' && !isMock && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#b45309] font-mono flex items-center gap-1">
              <AlertCircle className="w-3.5 h-3.5" />
              Viewing {date} (Reference only at 2019-01-01)
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

      {/* Summary Headline Cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-white border border-[#e2e8f0] rounded-xl p-3 shadow-xs">
            <span className="text-[11px] font-mono text-[#64748d] block">Peak Thermocline Error</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-lg font-mono font-bold text-[#0d253d]">
                {summary.peakRmse.rmse.toFixed(3)}
              </span>
              <span className="text-[11px] font-mono text-[#64748d]">°C RMSE</span>
            </div>
            <span className="text-[10px] font-mono text-[#94a3b8] mt-0.5 block">
              At {summary.peakRmse.depth_m}m level
            </span>
          </div>

          <div className="bg-white border border-[#e2e8f0] rounded-xl p-3 shadow-xs">
            <span className="text-[11px] font-mono text-[#64748d] block">Upper Ocean Skill</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-lg font-mono font-bold text-[#0284c7]">
                {summary.surface.rmse.toFixed(3)}
              </span>
              <span className="text-[11px] font-mono text-[#64748d]">°C RMSE</span>
            </div>
            <span className="text-[10px] font-mono text-[#94a3b8] mt-0.5 block">
              Surface (0m) layer
            </span>
          </div>

          <div className="bg-white border border-[#e2e8f0] rounded-xl p-3 shadow-xs">
            <span className="text-[11px] font-mono text-[#64748d] block">Abyssal Stability</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-lg font-mono font-bold text-[#10b981]">
                {summary.abyss.rmse.toFixed(3)}
              </span>
              <span className="text-[11px] font-mono text-[#64748d]">°C RMSE</span>
            </div>
            <span className="text-[10px] font-mono text-[#94a3b8] mt-0.5 block">
              1000m deep layer
            </span>
          </div>

          <div className="bg-white border border-[#e2e8f0] rounded-xl p-3 shadow-xs">
            <span className="text-[11px] font-mono text-[#64748d] block">Basin Column Mean Bias</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span
                className={`text-lg font-mono font-bold ${
                  summary.meanBias > 0
                    ? 'text-[#dc2626]'
                    : summary.meanBias < 0
                    ? 'text-[#2563eb]'
                    : 'text-[#0d253d]'
                }`}
              >
                {summary.meanBias >= 0 ? '+' : ''}
                {summary.meanBias.toFixed(3)}
              </span>
              <span className="text-[11px] font-mono text-[#64748d]">°C</span>
            </div>
            <span className="text-[10px] font-mono text-[#94a3b8] mt-0.5 block">
              Mean across 15 depths
            </span>
          </div>
        </div>
      )}

      {/* Main Analytical Section: Curve + Bias + Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: Primary Reconstruction Error vs Depth Plot (lg:col-span-7) */}
        <div className="lg:col-span-7 bg-white border border-[#e2e8f0] rounded-xl p-4 shadow-xs flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-[#f1f5f9]">
            <div className="flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-[#0284c7]" />
              <span className="text-xs font-semibold text-[#0d253d] uppercase tracking-wider">
                Reconstruction Error vs Depth
              </span>
            </div>

            <div className="flex items-center gap-4 text-[11px] font-mono">
              <span className="flex items-center gap-1.5">
                <span className="w-3.5 h-0.5 bg-[#0284c7] inline-block rounded-full" />
                <span className="text-[#0d253d] font-semibold">RMSE (Primary)</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3.5 h-0.5 border-t-2 border-dashed border-[#6366f1] inline-block" />
                <span className="text-[#64748d]">MAE (Secondary)</span>
              </span>
            </div>
          </div>

          {/* Canvas Rendering Plot */}
          <div className="relative mt-3 flex-1 min-h-[340px] flex items-center justify-center bg-white rounded-lg overflow-hidden border border-[#e2e8f0]">
            {loading && (
              <div className="absolute inset-0 z-10 bg-white/75 flex items-center justify-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin text-[#0284c7]" />
                <span className="text-xs font-mono text-[#0d253d]">Computing skill profile...</span>
              </div>
            )}

            {errorMsg ? (
              <div className="p-6 text-center text-xs font-mono text-[#b91c1c] max-w-md">
                <AlertCircle className="w-6 h-6 mx-auto mb-2 text-[#ef4444]" />
                {errorMsg}
              </div>
            ) : (
              <>
                <canvas
                  ref={canvasRef}
                  width={540}
                  height={350}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                  className="w-full h-auto max-h-[370px] cursor-crosshair block"
                />

                {/* Exact Metric Hover Tooltip */}
                {hoveredDepth !== null && hoverCoords && activeMetric && (
                  <div
                    className="absolute z-20 pointer-events-none bg-[#0d253d]/95 text-white px-2.5 py-1.5 rounded-md text-[11px] font-mono shadow-md border border-[#334155]/60 transition-all duration-75"
                    style={{
                      left: Math.min(hoverCoords.x + 14, 340),
                      top: Math.max(hoverCoords.y - 45, 12),
                    }}
                  >
                    <div className="font-bold text-[#f8fafc] border-b border-[#334155] pb-0.5 mb-1">
                      {hoveredDepth}m Level
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[#94a3b8]">RMSE:</span>
                      <span className="text-[#38bdf8] font-bold">{activeMetric.rmse.toFixed(3)}°C</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[#94a3b8]">MAE:</span>
                      <span className="text-[#c084fc] font-medium">{activeMetric.mae.toFixed(3)}°C</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[#94a3b8]">Mean Bias:</span>
                      <span
                        className={
                          activeMetric.mean_bias > 0
                            ? 'text-[#fca5a5] font-semibold'
                            : activeMetric.mean_bias < 0
                            ? 'text-[#93c5fd] font-semibold'
                            : 'text-[#e2e8f0]'
                        }
                      >
                        {activeMetric.mean_bias >= 0 ? '+' : ''}
                        {activeMetric.mean_bias.toFixed(3)}°C
                      </span>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Active Level Inspector Strip */}
          {activeMetric && (
            <div className="mt-3 pt-3 border-t border-[#f1f5f9] flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <div className="flex items-center gap-2">
                <Layers className="w-3.5 h-3.5 text-[#0284c7]" />
                <span className="font-bold text-[#0d253d]">{activeMetric.depth_m}m Level:</span>
              </div>
              <div className="flex items-center gap-4 text-[#64748d]">
                <span>
                  RMSE: <b className="text-[#0284c7]">{activeMetric.rmse.toFixed(3)}°C</b>
                </span>
                <span>
                  MAE: <b className="text-[#7c3aed]">{activeMetric.mae.toFixed(3)}°C</b>
                </span>
                <span>
                  Mean Bias:{' '}
                  <b
                    className={
                      activeMetric.mean_bias > 0
                        ? 'text-[#dc2626]'
                        : activeMetric.mean_bias < 0
                        ? 'text-[#2563eb]'
                        : 'text-[#0d253d]'
                    }
                  >
                    {activeMetric.mean_bias >= 0 ? '+' : ''}
                    {activeMetric.mean_bias.toFixed(3)}°C
                  </b>
                </span>
                <span className="text-[11px] text-[#94a3b8]">
                  {activeMetric.valid_cells.toLocaleString()} cells
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right: Mean Signed Bias by Depth + Full Matrix Table (lg:col-span-5) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Secondary: Mean Signed Bias by Depth Bar Chart */}
          <div className="bg-white border border-[#e2e8f0] rounded-xl p-4 shadow-xs">
            <div className="flex items-center justify-between pb-2 border-b border-[#f1f5f9]">
              <span className="text-xs font-semibold text-[#0d253d] uppercase tracking-wider">
                Mean Signed Bias by Depth (°C)
              </span>
              <span className="text-[11px] font-mono text-[#64748d]">
                Zero = Perfect Balance
              </span>
            </div>

            <div className="mt-2.5 space-y-1 max-h-[175px] overflow-y-auto pr-1">
              {metrics.map((m) => {
                const isSelected = m.depth_m === activeDepth;
                const bias = m.mean_bias;
                // Normalize bar: max expected bias is ~0.25°C
                const barWidth = Math.min(100, (Math.abs(bias) / 0.25) * 50);

                return (
                  <div
                    key={m.depth_m}
                    onClick={() => setSelectedDepth(m.depth_m)}
                    className={`flex items-center gap-2 py-0.5 px-1.5 rounded cursor-pointer transition-colors text-[11px] font-mono ${
                      isSelected ? 'bg-[#f0f9ff] font-bold' : 'hover:bg-[#f8fafc]'
                    }`}
                  >
                    <span className="w-10 text-right text-[#64748d] shrink-0">
                      {m.depth_m}m
                    </span>

                    {/* Diverging bar around 50% centerline */}
                    <div className="flex-1 h-3 bg-[#f1f5f9] rounded relative overflow-hidden flex items-center">
                      <div className="w-px h-full bg-[#cbd5e1] absolute left-1/2" />
                      {bias < 0 ? (
                        <div
                          className="h-2 bg-[#3b82f6] rounded-l absolute right-1/2"
                          style={{ width: `${barWidth}%` }}
                        />
                      ) : (
                        <div
                          className="h-2 bg-[#ef4444] rounded-r absolute left-1/2"
                          style={{ width: `${barWidth}%` }}
                        />
                      )}
                    </div>

                    <span
                      className={`w-14 text-right shrink-0 ${
                        bias > 0 ? 'text-[#dc2626]' : bias < 0 ? 'text-[#2563eb]' : 'text-[#64748d]'
                      }`}
                    >
                      {bias >= 0 ? '+' : ''}
                      {bias.toFixed(3)}°
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Depth Matrix Table */}
          <div className="bg-white border border-[#e2e8f0] rounded-xl p-4 shadow-xs">
            <div className="flex items-center justify-between pb-2 border-b border-[#f1f5f9]">
              <span className="text-xs font-semibold text-[#0d253d] uppercase tracking-wider">
                Skill Matrix (15 Standard Levels)
              </span>
              <span className="text-[11px] font-mono text-[#64748d]">
                Click row to inspect
              </span>
            </div>

            <div className="max-h-[220px] overflow-y-auto mt-2 border border-[#e2e8f0] rounded-lg">
              <table className="w-full text-left text-[11px] font-mono">
                <thead className="bg-[#f8fafc] text-[#64748d] sticky top-0 border-b border-[#e2e8f0]">
                  <tr>
                    <th className="py-1 px-2">Depth</th>
                    <th className="py-1 px-2 text-right">RMSE</th>
                    <th className="py-1 px-2 text-right">MAE</th>
                    <th className="py-1 px-2 text-right">Bias</th>
                    <th className="py-1 px-2 text-right">Cells</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#f1f5f9]">
                  {metrics.map((m) => {
                    const isSelected = m.depth_m === activeDepth;
                    return (
                      <tr
                        key={m.depth_m}
                        onClick={() => setSelectedDepth(m.depth_m)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-[#f0f9ff] font-semibold text-[#0d253d]' : 'hover:bg-[#f8fafc]'
                        }`}
                      >
                        <td className="py-1 px-2 text-[#0d253d]">
                          {m.depth_m}m {isSelected && '•'}
                        </td>
                        <td className="py-1 px-2 text-right text-[#0284c7] font-medium">
                          {m.rmse.toFixed(3)}°C
                        </td>
                        <td className="py-1 px-2 text-right text-[#7c3aed]">
                          {m.mae.toFixed(3)}°C
                        </td>
                        <td
                          className={`py-1 px-2 text-right ${
                            m.mean_bias > 0
                              ? 'text-[#dc2626]'
                              : m.mean_bias < 0
                              ? 'text-[#2563eb]'
                              : 'text-[#64748d]'
                          }`}
                        >
                          {m.mean_bias >= 0 ? '+' : ''}
                          {m.mean_bias.toFixed(3)}°
                        </td>
                        <td className="py-1 px-2 text-right text-[#94a3b8]">
                          {m.valid_cells.toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Scientific Footnote */}
      <div className="bg-[#f1f5f9]/70 border border-[#e2e8f0] rounded-lg p-3 text-[11px] font-mono text-[#64748d] flex items-start gap-2">
        <Info className="w-4 h-4 text-[#0284c7] shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-[#0d253d]">Scientific Integrity & Reference Note: </span>
          OceanIQ reconstruction error metrics are calculated against the GLORYS12V1 ocean reanalysis reference on the held-out 2019-01-01 test date across all valid ocean cells. Note: GLORYS12V1 is a numerical reanalysis product, not an independent in-situ observation. Independent in-situ observation comparisons are conducted separately using Argo profiling floats under the 'In-Situ Argo Validation' tab and are never merged with reanalysis metrics.
        </div>
      </div>
    </div>
  );
};
