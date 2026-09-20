import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import { Loader2 } from 'lucide-react';
import type { ReconstructionResponse } from '@/types/api';

interface ThermalSoundingPlotProps {
  reconstruction: ReconstructionResponse | null;
  loading: boolean;
  height?: number;
}

export const ThermalSoundingPlot: React.FC<ThermalSoundingPlotProps> = ({
  reconstruction,
  loading,
  height = 480,
}) => {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    if (!reconstruction) {
      // Clean empty state plot
      Plotly.newPlot(
        chartRef.current,
        [],
        {
          title: {
            text: 'Select coordinates and click "Reconstruct Profile"',
            font: { color: '#64748d', size: 12, family: 'var(--font-sans, system-ui)' },
          },
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
          xaxis: {
            title: { text: 'Temperature (°C)', font: { color: '#64748d', size: 11 } },
            color: '#64748d',
            range: [0, 35],
            gridcolor: '#f1f5f9',
            linecolor: '#e2e8f0',
          },
          yaxis: {
            title: { text: 'Depth (m)', font: { color: '#64748d', size: 11 } },
            autorange: 'reversed',
            color: '#64748d',
            range: [1000, 0],
            gridcolor: '#f1f5f9',
            linecolor: '#e2e8f0',
          },
          margin: { l: 54, r: 24, t: 36, b: 40 },
        },
        { responsive: true, displayModeBar: false }
      );
      return;
    }

    const traces: Plotly.Data[] = [
      {
        x: reconstruction.temperature_c,
        y: reconstruction.depths_m,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'OceanEmbed Reconstruction',
        line: { color: '#3b49df', width: 2.2, shape: 'spline', smoothing: 0.8 },
        marker: {
          color: '#3b49df',
          size: 6,
          symbol: 'circle',
          line: { color: '#ffffff', width: 1.5 },
        },
        hovertemplate:
          '<b>Depth:</b> %{y} m<br><b>Reconstructed T:</b> %{x:.2f} °C<extra></extra>',
      },
    ];

    // Collocated in-situ Argo float overlay when available
    if (reconstruction.argo_comparison) {
      const argo = reconstruction.argo_comparison;
      traces.push({
        x: argo.temperature_c,
        y: argo.depths_m,
        type: 'scatter',
        mode: 'lines+markers',
        name: `In-Situ Argo (${argo.float_id})`,
        line: { color: '#e11d48', width: 1.8, dash: 'dash', shape: 'spline', smoothing: 0.8 },
        marker: {
          color: '#e11d48',
          size: 6.5,
          symbol: 'diamond',
          line: { color: '#ffffff', width: 1.5 },
        },
        hovertemplate:
          '<b>Depth:</b> %{y} m<br><b>Argo Float T:</b> %{x:.2f} °C<extra></extra>',
      });
    }

    // D26 isotherm horizontal line
    const shapes: Partial<Plotly.Shape>[] = [];
    const annotations: Partial<Plotly.Annotation>[] = [];

    if (
      reconstruction.d26_depth_m !== null &&
      reconstruction.d26_depth_m !== undefined
    ) {
      shapes.push({
        type: 'line',
        x0: 2,
        x1: 34,
        y0: reconstruction.d26_depth_m,
        y1: reconstruction.d26_depth_m,
        line: { color: '#d97706', width: 1.2, dash: 'dot' },
      });
      annotations.push({
        x: 32,
        y: reconstruction.d26_depth_m,
        text: `D26: ${reconstruction.d26_depth_m}m`,
        showarrow: false,
        font: { color: '#92400e', size: 10, family: 'var(--font-mono, monospace)' },
        bgcolor: '#ffffff',
        bordercolor: '#e3e8ee',
        borderwidth: 1,
        borderpad: 3,
      });
    }

    const layout: Partial<Plotly.Layout> = {
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { color: '#0d253d', family: 'var(--font-sans, system-ui)', size: 11 },
      xaxis: {
        title: { text: 'Temperature (°C)', font: { size: 11, color: '#475569' } },
        range: [2, 34],
        gridcolor: '#f1f5f9',
        zerolinecolor: '#e2e8f0',
        linecolor: '#e2e8f0',
        tickfont: { size: 10, family: 'var(--font-mono, monospace)', color: '#64748d' },
      },
      yaxis: {
        title: { text: 'Depth (meters)', font: { size: 11, color: '#475569' } },
        autorange: 'reversed',
        gridcolor: '#f1f5f9',
        zerolinecolor: '#e2e8f0',
        linecolor: '#e2e8f0',
        tickvals: [0, 100, 200, 300, 500, 700, 1000],
        ticktext: ['0m', '100m', '200m', '300m', '500m', '700m', '1000m'],
        tickfont: { size: 10, family: 'var(--font-mono, monospace)', color: '#64748d' },
      },
      legend: {
        x: 0.35,
        y: 0.06,
        bgcolor: 'rgba(255, 255, 255, 0.95)',
        bordercolor: '#e3e8ee',
        borderwidth: 1,
        font: { size: 11, color: '#0d253d' },
      },
      shapes,
      annotations,
      margin: { l: 56, r: 24, t: 20, b: 44 },
      hovermode: 'closest',
    };

    Plotly.react(chartRef.current, traces, layout, {
      responsive: true,
      displayModeBar: false,
    });
  }, [reconstruction]);

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden bg-white">
      <div ref={chartRef} style={{ width: '100%', height: height !== undefined ? `${height}px` : '100%' }} />

      {loading && (
        <div className="absolute inset-0 bg-white/90 backdrop-blur-xs flex items-center justify-center text-[#533afd] font-mono text-sm gap-2 z-20">
          <Loader2 className="size-4 animate-spin" />
          <span className="font-medium text-[#0d253d]">Synthesizing 15-Depth Thermal Profile...</span>
        </div>
      )}
    </div>
  );
};

