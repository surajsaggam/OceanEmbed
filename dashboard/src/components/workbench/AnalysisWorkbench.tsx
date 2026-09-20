import React, { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { BasinLocationPicker } from '../map/BasinLocationPicker';
import { ThermalSoundingPlot } from '../profile/ThermalSoundingPlot';
import { ThermalMetricsMatrix } from '../profile/ThermalMetricsMatrix';
import { SurfaceDriversPanel } from '../surface/SurfaceDriversPanel';
import { LatentManifoldView } from '../manifold/LatentManifoldView';
import { ArgoValidationPanel } from '../validation/ArgoValidationPanel';
import { ScientificProvenanceCard } from '../audit/ScientificProvenanceCard';
import type {
  ReconstructionResponse,
  EmbeddingScatterResponse,
} from '@/types/api';

interface AnalysisWorkbenchProps {
  latitude: number;
  longitude: number;
  loading: boolean;
  reconstruction: ReconstructionResponse | null;
  scatterData: EmbeddingScatterResponse | null;
  onSelectCoordinates: (lat: number, lon: number) => void;
}

type SoundingViewMode = 'chart' | 'split' | 'table';

export const AnalysisWorkbench: React.FC<AnalysisWorkbenchProps> = ({
  latitude,
  longitude,
  loading,
  reconstruction,
  scatterData,
  onSelectCoordinates,
}) => {
  const [viewMode, setViewMode] = useState<SoundingViewMode>('split');
  const [diagnosticTab, setDiagnosticTab] = useState<string>('drivers');

  const argo = reconstruction?.argo_comparison;

  return (
    <main className="max-w-[1640px] w-full mx-auto px-5 lg:px-6 pt-4 pb-6 space-y-4">
      {/* Upper Section: Spatial Geographic Anchor (Left) + Centerpiece Thermal Sounding (Right) */}
      <section
        className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start"
        aria-label="Targeting Map and Primary Subsurface Sounding"
      >
        {/* Spatial Geographic Anchor (Leaflet Basin Map) */}
        <div className="lg:col-span-5 flex flex-col gap-1.5">
          <div className="flex items-center justify-between px-1">
            <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
              North Indian Ocean Basin
            </span>
            <span className="text-[13px] font-mono text-[#64748d]">
              Click basin to snap (0.25°)
            </span>
          </div>

          <BasinLocationPicker
            latitude={latitude}
            longitude={longitude}
            onSelectCoordinates={onSelectCoordinates}
            height="calc(100vh - 220px)"
          />
        </div>

        {/* Centerpiece Analytical Stage (Plotly 15-Depth Thermal Sounding) */}
        <div className="lg:col-span-7 flex flex-col rounded-xl border border-[#e3e8ee] bg-white overflow-hidden shadow-xs">
          {/* Sounding Stage Header */}
          <div className="px-5 py-3 bg-white border-b border-[#e2e8f0] flex flex-wrap items-center justify-between gap-3">
            <div>
              <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider block">
                Subsurface Temperature Profile
              </span>
              <span className="text-[13px] font-mono text-[#64748d]">
                15 Standard Levels (0–1000m)
              </span>
            </div>

            {/* Diagnostic Metrics Readout (Quiet scientific text, NO badge pills) */}
            {reconstruction && (
              <div className="flex items-center gap-3 font-mono text-sm text-[#64748d]">
                {reconstruction.d26_depth_m !== null && reconstruction.d26_depth_m !== undefined && (
                  <span title="Depth of 26°C isotherm — Tropical Cyclone Heat Potential proxy">
                    D26: <strong className="text-[#b45309] font-medium tabular-nums">{reconstruction.d26_depth_m}m</strong>
                  </span>
                )}

                {reconstruction.mixed_layer_depth_m !== null && reconstruction.mixed_layer_depth_m !== undefined && (
                  <span title="Mixed Layer Depth threshold = 0.2°C">
                    MLD: <strong className="text-[#0284c7] font-medium tabular-nums">{reconstruction.mixed_layer_depth_m}m</strong>
                  </span>
                )}

                {argo && argo.rmse !== null && argo.rmse !== undefined && (
                  <span title="RMSE vs collocated synthetic Argo float profile">
                    Argo RMSE: <strong className="text-[#e11d48] font-medium tabular-nums">{argo.rmse.toFixed(2)}°C</strong>
                  </span>
                )}
              </div>
            )}

            {/* View Mode Segmented Controls (Apple-style pill tabs) */}
            <div className="flex items-center p-0.5 rounded-full bg-[#f1f5f9] border border-[#e2e8f0] text-sm" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === 'chart'}
                onClick={() => setViewMode('chart')}
                className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                  viewMode === 'chart'
                    ? 'bg-white text-[#0d253d] font-medium shadow-xs'
                    : 'text-[#64748d] hover:text-[#0d253d]'
                }`}
              >
                Profile Curve
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={viewMode === 'split'}
                onClick={() => setViewMode('split')}
                className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                  viewMode === 'split'
                    ? 'bg-white text-[#0d253d] font-medium shadow-xs'
                    : 'text-[#64748d] hover:text-[#0d253d]'
                }`}
              >
                Split View
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={viewMode === 'table'}
                onClick={() => setViewMode('table')}
                className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                  viewMode === 'table'
                    ? 'bg-white text-[#0d253d] font-medium shadow-xs'
                    : 'text-[#64748d] hover:text-[#0d253d]'
                }`}
              >
                Depth Matrix
              </button>
            </div>
          </div>

          {/* Sounding Content Area — viewport-relative height, stays within one screen */}
          <div className="p-3 bg-white" style={{ height: 'calc(100vh - 220px - 66px)' }}>
            {viewMode === 'chart' && (
              <div className="w-full h-full">
                <ThermalSoundingPlot
                  reconstruction={reconstruction}
                  loading={loading}
                  height={undefined}
                />
              </div>
            )}

            {viewMode === 'split' && (
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 h-full">
                <div className="xl:col-span-7 h-full">
                  <ThermalSoundingPlot
                    reconstruction={reconstruction}
                    loading={loading}
                    height={undefined}
                  />
                </div>
                <div className="xl:col-span-5 h-full flex flex-col overflow-auto">
                  <ThermalMetricsMatrix
                    reconstruction={reconstruction}
                    maxHeight="100%"
                  />
                </div>
              </div>
            )}

            {viewMode === 'table' && (
              <div className="w-full h-full overflow-auto">
                <ThermalMetricsMatrix
                  reconstruction={reconstruction}
                  maxHeight="100%"
                />
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Lower Section: Diagnostic Analysis & Provenance (Single White Panel) */}
      <section
        className="rounded-xl border border-[#e3e8ee] bg-white p-5 shadow-xs"
        aria-label="Secondary Diagnostic Analysis"
      >
        <Tabs value={diagnosticTab} onValueChange={setDiagnosticTab}>
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#e2e8f0] pb-3">
            <div>
              <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider block">
                Diagnostic Analysis & Scientific Lineage
              </span>
              <span className="text-[13px] text-[#64748d] font-mono">
                Multimodal observation inversion, latent representation, and independent ground truth
              </span>
            </div>

            <TabsList className="h-8">
              <TabsTrigger value="drivers" className="text-sm px-3">
                Surface Observations (7)
              </TabsTrigger>

              <TabsTrigger value="manifold" className="text-sm px-3">
                128-D Latent Manifold
              </TabsTrigger>

              <TabsTrigger value="validation" className="text-sm px-3">
                In-Situ Argo Validation
              </TabsTrigger>

              <TabsTrigger value="provenance" className="text-sm px-3">
                Model Lineage & Audit
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="drivers" className="pt-4 m-0">
            <SurfaceDriversPanel
              surface={reconstruction?.surface_context ?? null}
              isMock={reconstruction?.is_mock ?? true}
            />
          </TabsContent>

          <TabsContent value="manifold" className="pt-4 m-0">
            <LatentManifoldView
              scatterData={scatterData}
              activeEmbedding={reconstruction?.embedding ?? null}
              height={340}
            />
          </TabsContent>

          <TabsContent value="validation" className="pt-4 m-0">
            <ArgoValidationPanel argo={reconstruction?.argo_comparison} />
          </TabsContent>

          <TabsContent value="provenance" className="pt-4 m-0">
            <ScientificProvenanceCard reconstruction={reconstruction} />
          </TabsContent>
        </Tabs>
      </section>
    </main>
  );
};

