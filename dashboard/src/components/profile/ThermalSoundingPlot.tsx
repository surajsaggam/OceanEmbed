import React, { useEffect, useRef, useState } from 'react';
import {
  createOptionsChart,
  AreaSeries,
  LineSeries,
  ColorType,
  LineStyle,
  CrosshairMode,
  type IChartApiBase,
  type ISeriesApi,
} from 'lightweight-charts';
import { Loader2 } from 'lucide-react';
import type { ReconstructionResponse } from '@/types/api';

interface ThermalSoundingPlotProps {
  reconstruction: ReconstructionResponse | null;
  loading: boolean;
  height?: number;
}

interface HoveredPoint {
  depth: number;
  reconTemp: number | null;
  argoTemp: number | null;
}

export const ThermalSoundingPlot: React.FC<ThermalSoundingPlotProps> = ({
  reconstruction,
  loading,
  height,
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<HoveredPoint | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApiBase<number> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Clean previous chart instance if existing
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
      chartInstanceRef.current = null;
    }

    const chart = createOptionsChart(containerRef.current, {
      autoSize: true,
      layout: {
        attributionLogo: false,
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#64748d',
        fontFamily: 'var(--font-sans, system-ui)',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#f8fafc', style: LineStyle.Solid },
        horzLines: { color: '#f1f5f9', style: LineStyle.Solid },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#94a3b8',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#0d253d',
        },
        horzLine: {
          color: '#94a3b8',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#0d253d',
        },
      },
      rightPriceScale: {
        borderColor: '#e2e8f0',
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: '#e2e8f0',
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      handleScroll: false,
      handleScale: {
        axisPressedMouseMove: false,
        mouseWheel: false,
        pinch: false,
      },
      localization: {
        precision: 2,
        priceFormatter: (price: number) => `${price.toFixed(2)} °C`,
        timeFormatter: (depth: number) => `${depth}m`,
      },
    });

    chartInstanceRef.current = chart;

    if (!reconstruction) {
      return;
    }

    // Add OceanEmbed Reconstruction Area Series
    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: '#3b49df',
      topColor: 'rgba(59, 73, 223, 0.20)',
      bottomColor: 'rgba(59, 73, 223, 0.02)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => `${price.toFixed(2)} °C`,
      },
    });

    const reconData = reconstruction.depths_m.map((depth, idx) => ({
      time: depth,
      value: reconstruction.temperature_c[idx],
    }));
    areaSeries.setData(reconData);

    // D26 isotherm horizontal reference line
    if (
      reconstruction.d26_depth_m !== null &&
      reconstruction.d26_depth_m !== undefined &&
      reconstruction.d26_depth_m > 0
    ) {
      areaSeries.createPriceLine({
        price: 26.0,
        color: '#d97706',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `D26 (${reconstruction.d26_depth_m}m)`,
      });
    }

    // Collocated in-situ Argo float overlay series
    let argoSeries: ISeriesApi<'Line', number> | null = null;
    if (reconstruction.argo_comparison) {
      const argo = reconstruction.argo_comparison;
      argoSeries = chart.addSeries(LineSeries, {
        color: '#e11d48',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
        lastValueVisible: false,
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => `${price.toFixed(2)} °C`,
        },
      });

      const argoData = argo.depths_m.map((depth, idx) => ({
        time: depth,
        value: argo.temperature_c[idx],
      }));
      argoSeries.setData(argoData);
    }

    chart.timeScale().fitContent();

    // Crosshair hover synchronization with HUD
    chart.subscribeCrosshairMove((param) => {
      if (!param.time || param.point === undefined) {
        setHoveredPoint(null);
        return;
      }
      const depth = param.time as number;
      const recIndex = reconstruction.depths_m.indexOf(depth);
      const recVal = recIndex >= 0 ? reconstruction.temperature_c[recIndex] : null;

      let argoVal: number | null = null;
      if (reconstruction.argo_comparison) {
        const argoIndex = reconstruction.argo_comparison.depths_m.indexOf(depth);
        if (argoIndex >= 0) {
          argoVal = reconstruction.argo_comparison.temperature_c[argoIndex];
        }
      }

      setHoveredPoint({
        depth,
        reconTemp: recVal,
        argoTemp: argoVal,
      });
    });

    return () => {
      chart.remove();
      chartInstanceRef.current = null;
    };
  }, [reconstruction]);

  // Ensure chart fits content on size updates
  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.timeScale().fitContent();
    }
  }, [height]);

  const containerStyle = height !== undefined ? { height: `${height}px` } : { height: '100%' };

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden bg-white flex flex-col" style={containerStyle}>
      {/* Top Header Bar: Live HUD & Essential Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[#f1f5f9] bg-white z-10 flex-shrink-0">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {hoveredPoint ? (
            <div className="flex items-center gap-3 font-mono">
              <span className="text-[#64748d]">
                Depth: <strong className="text-[#0d253d] font-semibold">{hoveredPoint.depth}m</strong>
              </span>
              {hoveredPoint.reconTemp !== null && (
                <span className="text-[#3b49df]">
                  Recon: <strong>{hoveredPoint.reconTemp.toFixed(2)}°C</strong>
                </span>
              )}
              {hoveredPoint.argoTemp !== null && (
                <span className="text-[#e11d48]">
                  Argo: <strong>{hoveredPoint.argoTemp.toFixed(2)}°C</strong>
                </span>
              )}
              {hoveredPoint.reconTemp !== null && hoveredPoint.argoTemp !== null && (
                <span className="text-[#64748d]">
                  ΔT: <strong className={Math.abs(hoveredPoint.reconTemp - hoveredPoint.argoTemp) < 0.05 ? 'text-[#64748d]' : hoveredPoint.reconTemp > hoveredPoint.argoTemp ? 'text-[#b45309]' : 'text-[#0284c7]'}>
                    {(hoveredPoint.reconTemp - hoveredPoint.argoTemp > 0 ? '+' : '') +
                      (hoveredPoint.reconTemp - hoveredPoint.argoTemp).toFixed(2)}°C
                  </strong>
                </span>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-3 font-mono text-[#64748d]">
              <span className="flex items-center gap-1.5 font-sans font-medium text-[#0d253d]">
                <span className="w-2.5 h-0.5 bg-[#3b49df] rounded-full inline-block" />
                OceanEmbed Recon
              </span>
              {reconstruction?.argo_comparison && (
                <span className="flex items-center gap-1.5 font-sans font-medium text-[#e11d48]">
                  <span className="w-2.5 h-0.5 border-b border-dashed border-[#e11d48] inline-block" />
                  In-Situ Argo ({reconstruction.argo_comparison.float_id})
                </span>
              )}
              {reconstruction?.d26_depth_m !== undefined && (
                <span className="flex items-center gap-1.5 text-[#b45309]">
                  <span className="w-2 h-2 rounded-full bg-[#d97706]/20 border border-[#d97706] inline-block" />
                  D26: {reconstruction.d26_depth_m && reconstruction.d26_depth_m > 0 ? `${reconstruction.d26_depth_m}m` : 'Not reached'}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main Chart Viewport */}
      <div className="relative flex-1 w-full min-h-0">
        {/* TradingView Chart Container */}
        <div ref={containerRef} className="w-full h-full" />

        {/* Empty State Overlay if no reconstruction is loaded */}
        {!reconstruction && !loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center bg-white/95 z-10 pointer-events-none">
            <div className="text-sm font-semibold text-[#0d253d] mb-1">
              No Reconstruction Profile Loaded
            </div>
            <div className="text-xs text-[#64748d] max-w-sm leading-relaxed">
              Select coordinates on the map and click &ldquo;Reconstruct Profile&rdquo; to inspect the subsurface thermal sounding.
            </div>
          </div>
        )}

        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-white/90 backdrop-blur-xs flex items-center justify-center text-[#533afd] font-mono text-sm gap-2 z-20">
            <Loader2 className="size-4 animate-spin" />
            <span className="font-medium text-[#0d253d]">Synthesizing 15-Depth Thermal Profile...</span>
          </div>
        )}
      </div>
    </div>
  );
};
