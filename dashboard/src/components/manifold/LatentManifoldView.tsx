import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { groupScatterPointsByRegime } from '@/lib/ocean';
import type { EmbeddingScatterResponse, EmbeddingCoordinates } from '@/types/api';

interface LatentManifoldViewProps {
  scatterData: EmbeddingScatterResponse | null;
  activeEmbedding: EmbeddingCoordinates | null;
  height?: number;
}

export const LatentManifoldView: React.FC<LatentManifoldViewProps> = ({
  scatterData,
  activeEmbedding,
  height = 340,
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    if (!chartInstanceRef.current) {
      chartInstanceRef.current = echarts.init(chartRef.current, undefined, {
        renderer: 'canvas',
      });
    }

    const chart = chartInstanceRef.current;

    if (!scatterData || scatterData.points.length === 0) {
      chart.setOption({
        title: {
          text: 'No Latent Manifold Data Available',
          left: 'center',
          top: 'center',
          textStyle: { color: '#64748d', fontSize: 14, fontFamily: 'var(--font-sans, system-ui)' },
        },
      });
      return;
    }

    const grouped = groupScatterPointsByRegime(scatterData.points);
    const regimes = Object.keys(grouped);

    // Harmonious curated palette on light canvas
    const regimeColors = [
      '#0284c7', '#3b82f6', '#6366f1', '#8b5cf6',
      '#ec4899', '#f43f5e', '#d97706', '#059669', '#0d9488'
    ];

    const series: echarts.SeriesOption[] = regimes.map((regime, idx) => {
      const color = regimeColors[idx % regimeColors.length];

      return {
        name: regime,
        type: 'scatter',
        data: grouped[regime].map((p) => ({
          value: [p.pca_1, p.pca_2],
          itemData: p,
        })),
        symbolSize: 6,
        itemStyle: {
          color,
          opacity: 0.75,
        },
        emphasis: {
          focus: 'series',
          itemStyle: {
            opacity: 1,
            borderColor: '#0d253d',
            borderWidth: 1.5,
          },
        },
      };
    });

    // Active station target series overlay
    if (activeEmbedding) {
      series.push({
        name: 'Target Station',
        type: 'scatter',
        data: [
          {
            value: [activeEmbedding.pca_1, activeEmbedding.pca_2],
            name: activeEmbedding.regime_label,
          } as any,
        ],
        symbol: 'diamond',
        symbolSize: 14,
        z: 10,
        itemStyle: {
          color: '#533afd',
          borderColor: '#ffffff',
          borderWidth: 2,
          shadowBlur: 6,
          shadowColor: 'rgba(83, 58, 253, 0.4)',
        },
        label: {
          show: true,
          formatter: '★ Target Station',
          position: 'top',
          color: '#533afd',
          fontSize: 11,
          fontFamily: 'var(--font-mono, monospace)',
          fontWeight: 'bold',
        },
      });
    }

    const option: echarts.EChartsOption = {
      backgroundColor: '#ffffff',
      grid: {
        left: 45,
        right: 25,
        top: 30,
        bottom: 30,
      },
      tooltip: {
        trigger: 'item',
        axisPointer: {
          show: true,
          type: 'cross',
          lineStyle: {
            color: '#94a3b8',
            width: 1,
            type: 'dashed',
          },
          label: {
            backgroundColor: '#0d253d',
            color: '#ffffff',
            fontSize: 11,
            fontFamily: 'var(--font-mono, monospace)',
            borderRadius: 4,
            padding: [3, 6],
          },
        },
        backgroundColor: 'rgba(255, 255, 255, 0.98)',
        borderColor: '#e2e8f0',
        textStyle: { color: '#0d253d', fontSize: 13 },
        extraCssText: 'box-shadow: 0 4px 12px rgba(0, 55, 112, 0.08); border-radius: 8px;',
        formatter: (params: any) => {
          if (params.seriesName === 'Target Station') {
            return `<b>Active Target Station</b><br/>Region / Basin: ${activeEmbedding?.regime_label}<br/>PCA Coordinates: [${params.value[0].toFixed(2)}, ${params.value[1].toFixed(2)}]`;
          }
          const item = params.data.itemData;
          if (!item) return '';
          return `<b>${item.regime}</b><br/>Region: ${item.region} (${item.season})<br/>Coords: ${item.latitude}°N, ${item.longitude}°E<br/>PCA: [${item.pca_1.toFixed(2)}, ${item.pca_2.toFixed(2)}]`;
        },
      },
      xAxis: {
        type: 'value',
        name: 'PCA 1',
        nameLocation: 'end',
        nameTextStyle: { color: '#64748d', fontSize: 12 },
        splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
        axisLabel: { color: '#64748d', fontSize: 12, fontFamily: 'var(--font-mono, monospace)' },
        axisLine: { lineStyle: { color: '#e2e8f0' } },
      },
      yAxis: {
        type: 'value',
        name: 'PCA 2',
        nameLocation: 'end',
        nameTextStyle: { color: '#64748d', fontSize: 12 },
        splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
        axisLabel: { color: '#64748d', fontSize: 12, fontFamily: 'var(--font-mono, monospace)' },
        axisLine: { lineStyle: { color: '#e2e8f0' } },
      },
      series,
    };

    chart.setOption(option, true);

    const handleResize = () => chart.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [scatterData, activeEmbedding]);

  return (
    <div className="space-y-3" role="region" aria-label="128-D Latent Ocean Embedding Manifold">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#e2e8f0] pb-2">
        <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
          128-D Latent Bottleneck Manifold (Principal Component Projection)
        </span>
        <span className="text-[13px] font-mono text-[#64748d]">
          Latent State: Z ∈ ℝ¹²⁸ · Subspace: PC1 vs PC2
        </span>
      </div>

      <div className="rounded-xl border border-[#e3e8ee] bg-white p-2.5 shadow-xs">
        <div ref={chartRef} style={{ width: '100%', height: `${height}px` }} />
      </div>

      <div className="flex items-center justify-between text-sm text-[#64748d] font-mono px-1">
        <span>Subspace: Linear Dimensionality Reduction (PCA 1 vs PCA 2)</span>
        {activeEmbedding && (
          <span className="text-[#3b49df] font-medium">
            Region / Basin: {activeEmbedding.regime_label}
          </span>
        )}
      </div>
    </div>
  );
};
