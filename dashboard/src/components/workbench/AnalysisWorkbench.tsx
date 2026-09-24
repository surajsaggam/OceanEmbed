import React, { useState, useEffect, useCallback } from 'react';
import { Maximize2 } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { BasinLocationPicker } from '../map/BasinLocationPicker';
import { ThermalSoundingPlot } from '../profile/ThermalSoundingPlot';
import { ThermalMetricsMatrix } from '../profile/ThermalMetricsMatrix';
import { VerticalTransectPlot } from '../profile/VerticalTransectPlot';
import { SurfaceDriversPanel } from '../surface/SurfaceDriversPanel';
import { LatentManifoldView } from '../manifold/LatentManifoldView';
import { ArgoValidationPanel } from '../validation/ArgoValidationPanel';
import { ScientificProvenanceCard } from '../audit/ScientificProvenanceCard';
import { ReconstructionDeparturePanel } from '../departure/ReconstructionDeparturePanel';
import { DepthWiseSkillProfilePanel } from '../skill/DepthWiseSkillProfilePanel';
import { fetchTransect } from '@/services/api';
import type {
  ReconstructionResponse,
  EmbeddingScatterResponse,
  TransectResponse,
  TransectPoint,
} from '@/types/api';

interface AnalysisWorkbenchProps {
  date?: string;
  latitude: number;
  longitude: number;
  loading: boolean;
  reconstruction: ReconstructionResponse | null;
  scatterData: EmbeddingScatterResponse | null;
  onSelectCoordinates: (lat: number, lon: number) => void;
  onSelectDate?: (date: string) => void;
}

type SoundingViewMode = 'chart' | 'split' | 'table' | 'transect';

export const AnalysisWorkbench: React.FC<AnalysisWorkbenchProps> = ({
  date,
  latitude,
  longitude,
  loading,
  reconstruction,
  scatterData,
  onSelectCoordinates,
  onSelectDate,
}) => {
  const [viewMode, setViewMode] = useState<SoundingViewMode>('split');
  const [diagnosticTab, setDiagnosticTab] = useState<string>('drivers');
  const [isProfileExpanded, setIsProfileExpanded] = useState<boolean>(false);

  // Transect State
  const [transectMode, setTransectMode] = useState<boolean>(false);
  const [transectPoints, setTransectPoints] = useState<TransectPoint[]>([
    { latitude: 15.0, longitude: 58.0 },
    { latitude: 15.0, longitude: 73.0 },
  ]);
  const [transectData, setTransectData] = useState<TransectResponse | null>(null);
  const [transectLoading, setTransectLoading] = useState<boolean>(false);

  const effectiveDate = date || reconstruction?.date || '2019-01-01';

  const loadTransect = useCallback(
    async (pointsToFetch: TransectPoint[], fetchDate = effectiveDate) => {
      if (pointsToFetch.length < 2) return;
      setTransectLoading(true);
      try {
        const res = await fetchTransect({
          date: fetchDate,
          points: pointsToFetch,
          num_samples: 30,
        });
        setTransectData(res);
      } catch (err) {
        console.error('Failed to load vertical transect:', err);
      } finally {
        setTransectLoading(false);
      }
    },
    [effectiveDate]
  );

  // Pre-load initial transect on mount or date change
  useEffect(() => {
    if (transectPoints.length >= 2) {
      loadTransect(transectPoints, effectiveDate);
    }
  }, [effectiveDate]);

  const handleAddTransectPoint = (lat: number, lon: number) => {
    if (transectPoints.length >= 2) {
      // Start fresh transect
      const newPts = [{ latitude: lat, longitude: lon }];
      setTransectPoints(newPts);
      setTransectData(null);
    } else {
      // Complete 2-point transect
      const newPts = [...transectPoints, { latitude: lat, longitude: lon }];
      setTransectPoints(newPts);
      loadTransect(newPts);
    }
  };

  const handleClearTransect = () => {
    setTransectPoints([]);
    setTransectData(null);
  };

  const handleSelectPresetTransect = (presetPoints: TransectPoint[]) => {
    setTransectPoints(presetPoints);
    loadTransect(presetPoints);
  };

  const handleToggleTransectMode = () => {
    const nextMode = !transectMode;
    setTransectMode(nextMode);
    if (nextMode) {
      setViewMode('transect');
    }
  };

  const argo = reconstruction?.argo_comparison;

  return (
    <main className="max-w-[1640px] w-full mx-auto px-5 lg:px-6 pt-4 pb-6 space-y-4">
      {/* Primary Analytical Horizon (Two Master Cards side-by-side) */}
      <section
        className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start"
        aria-label="Primary Analytical Horizon"
      >
        {/* Interactive NIO Geospatial Reference Stage */}
        <div
          className="lg:col-span-5 flex flex-col rounded-xl border border-[#e3e8ee] bg-white overflow-hidden shadow-xs"
          style={{ height: 'calc(100vh - 235px)', minHeight: '410px', maxHeight: '490px' }}
        >
          <div className="px-5 py-3 bg-white border-b border-[#e2e8f0] flex items-center justify-between shrink-0">
            <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
              North Indian Ocean Basin
            </span>
            <span className="text-[13px] font-mono text-[#64748d]">
              {transectMode ? 'Click 2 points for Transect' : 'Click basin to snap (0.25°)'}
            </span>
          </div>

          <div className="flex-1 min-h-0 w-full relative">
            <BasinLocationPicker
              latitude={latitude}
              longitude={longitude}
              onSelectCoordinates={onSelectCoordinates}
              height="100%"
              transectMode={transectMode || viewMode === 'transect'}
              transectPoints={transectPoints}
              onAddTransectPoint={handleAddTransectPoint}
              onClearTransect={handleClearTransect}
              onToggleTransectMode={handleToggleTransectMode}
            />
          </div>
        </div>

        {/* Centerpiece Analytical Stage (Subsurface Temperature Profile / Transect) */}
        <div
          className="lg:col-span-7 flex flex-col rounded-xl border border-[#e3e8ee] bg-white overflow-hidden shadow-xs"
          style={{ height: 'calc(100vh - 235px)', minHeight: '410px', maxHeight: '490px' }}
        >
          {/* Sounding Stage Header */}
          <div className="px-5 py-3 bg-white border-b border-[#e2e8f0] flex flex-wrap items-center justify-between gap-3 shrink-0">
            <div>
              <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider block">
                {viewMode === 'transect'
                  ? 'Vertical Subsurface Transect (Cross-Section)'
                  : 'Subsurface Temperature Profile'}
              </span>
              <span className="text-[13px] font-mono text-[#64748d]">
                15 Standard Levels (0–1000m)
              </span>
            </div>

            {/* Diagnostic Metrics Readout */}
            {reconstruction && viewMode !== 'transect' && (
              <div className="flex items-center gap-3 font-mono text-sm text-[#64748d]">
                {reconstruction.d26_depth_m !== undefined && (
                  <span title="Depth of 26°C isotherm — Tropical Cyclone Heat Potential proxy">
                    D26: <strong className="text-[#b45309] font-medium tabular-nums">{reconstruction.d26_depth_m && reconstruction.d26_depth_m > 0 ? `${reconstruction.d26_depth_m}m` : 'Not reached'}</strong>
                  </span>
                )}

                {reconstruction.mixed_layer_depth_m !== null && reconstruction.mixed_layer_depth_m !== undefined && (
                  <span title="Mixed Layer Depth threshold = 0.2°C">
                    MLD: <strong className="text-[#0284c7] font-medium tabular-nums">{reconstruction.mixed_layer_depth_m}m</strong>
                  </span>
                )}

                {argo && argo.rmse !== null && argo.rmse !== undefined && (
                  <span title="RMSE vs collocated in-situ Argo float profile">
                    Argo RMSE: <strong className="text-[#e11d48] font-medium tabular-nums">{argo.rmse.toFixed(2)}°C</strong>
                  </span>
                )}
              </div>
            )}

            {/* View Mode Segmented Controls + Expand Button */}
            <div className="flex items-center gap-2">
              <div className="flex items-center p-0.5 rounded-full bg-[#f1f5f9] border border-[#e2e8f0] text-sm" role="tablist">
                <button
                  type="button"
                  role="tab"
                  aria-selected={viewMode === 'chart'}
                  onClick={() => {
                    setViewMode('chart');
                    setTransectMode(false);
                  }}
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
                  onClick={() => {
                    setViewMode('split');
                    setTransectMode(false);
                  }}
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
                  onClick={() => {
                    setViewMode('table');
                    setTransectMode(false);
                  }}
                  className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                    viewMode === 'table'
                      ? 'bg-white text-[#0d253d] font-medium shadow-xs'
                      : 'text-[#64748d] hover:text-[#0d253d]'
                  }`}
                >
                  Depth Matrix
                </button>

                <button
                  type="button"
                  role="tab"
                  aria-selected={viewMode === 'transect'}
                  onClick={() => {
                    setViewMode('transect');
                    setTransectMode(true);
                    if (!transectData && transectPoints.length >= 2) {
                      loadTransect(transectPoints);
                    }
                  }}
                  className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                    viewMode === 'transect'
                      ? 'bg-white text-[#4338ca] font-semibold shadow-xs'
                      : 'text-[#64748d] hover:text-[#0d253d]'
                  }`}
                >
                  Vertical Transect
                </button>
              </div>

              {/* Expand to Dialog Button */}
              <button
                type="button"
                onClick={() => setIsProfileExpanded(true)}
                className="p-1.5 rounded-md border border-[#e2e8f0] bg-white hover:bg-[#f8fafc] text-[#64748d] hover:text-[#0d253d] transition-colors cursor-pointer flex items-center justify-center shadow-xs"
                title="Expand View"
                aria-label="Expand View"
              >
                <Maximize2 className="size-4" />
              </button>
            </div>
          </div>

          {/* Sounding Content Area */}
          <div className="flex-1 min-h-0 w-full p-3 bg-white overflow-hidden">
            {viewMode === 'chart' && (
              <div className="w-full h-full min-h-0">
                <ThermalSoundingPlot
                  reconstruction={reconstruction}
                  loading={loading}
                  height={undefined}
                />
              </div>
            )}

            {viewMode === 'split' && (
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-3 h-full min-h-0">
                <div className="xl:col-span-7 h-full min-w-0 min-h-0">
                  <ThermalSoundingPlot
                    reconstruction={reconstruction}
                    loading={loading}
                    height={undefined}
                  />
                </div>
                <div className="xl:col-span-5 h-full min-w-0 min-h-0 flex flex-col overflow-hidden">
                  <ThermalMetricsMatrix
                    reconstruction={reconstruction}
                    maxHeight="100%"
                  />
                </div>
              </div>
            )}

            {viewMode === 'table' && (
              <div className="w-full h-full min-h-0 overflow-hidden flex flex-col">
                <ThermalMetricsMatrix
                  reconstruction={reconstruction}
                  maxHeight="100%"
                />
              </div>
            )}

            {viewMode === 'transect' && (
              <div className="w-full h-full min-h-0 overflow-hidden flex flex-col">
                <VerticalTransectPlot
                  transect={transectData}
                  loading={transectLoading}
                  onSelectPreset={handleSelectPresetTransect}
                  onClearTransect={handleClearTransect}
                />
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Expanded Subsurface Temperature Profile Dialog */}
      <Dialog open={isProfileExpanded} onOpenChange={setIsProfileExpanded}>
        <DialogContent className="w-[96vw] max-w-[96vw] sm:max-w-[96vw] lg:max-w-[1520px] xl:max-w-[1640px] 2xl:max-w-[1760px] h-[90vh] max-h-[92vh] flex flex-col p-6 bg-white border-[#e3e8ee] text-[#0d253d] shadow-2xl rounded-2xl overflow-hidden">
          <DialogHeader className="flex-shrink-0">
            <div className="flex flex-wrap items-center justify-between gap-3 pr-8 pb-3 border-b border-[#e2e8f0]">
              <div>
                <DialogTitle className="text-base font-semibold text-[#0d253d] uppercase tracking-wider block">
                  {viewMode === 'transect'
                    ? 'Vertical Subsurface Transect (Cross-Section)'
                    : 'Subsurface Temperature Profile'}
                </DialogTitle>
                <DialogDescription className="text-[13px] font-mono text-[#64748d]">
                  15 Standard Levels (0–1000m) · Expanded Analytical View
                </DialogDescription>
              </div>

              {/* View Mode Segmented Controls in Dialog */}
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

                <button
                  type="button"
                  role="tab"
                  aria-selected={viewMode === 'transect'}
                  onClick={() => {
                    setViewMode('transect');
                    setTransectMode(true);
                    if (!transectData && transectPoints.length >= 2) {
                      loadTransect(transectPoints);
                    }
                  }}
                  className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                    viewMode === 'transect'
                      ? 'bg-white text-[#4338ca] font-semibold shadow-xs'
                      : 'text-[#64748d] hover:text-[#0d253d]'
                  }`}
                >
                  Vertical Transect
                </button>
              </div>
            </div>
          </DialogHeader>

          {/* Dialog Analytical Body */}
          <div className="flex-1 min-h-0 w-full pt-4 overflow-hidden">
            {viewMode === 'chart' && (
              <div className="w-full h-full min-h-0">
                <ThermalSoundingPlot
                  reconstruction={reconstruction}
                  loading={loading}
                  height={undefined}
                />
              </div>
            )}

            {viewMode === 'split' && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full min-h-0">
                <div className="lg:col-span-7 h-full min-h-0 min-w-0">
                  <ThermalSoundingPlot
                    reconstruction={reconstruction}
                    loading={loading}
                    height={undefined}
                  />
                </div>
                <div className="lg:col-span-5 h-full min-h-0 min-w-0 flex flex-col overflow-hidden">
                  <ThermalMetricsMatrix
                    reconstruction={reconstruction}
                    maxHeight="100%"
                  />
                </div>
              </div>
            )}

            {viewMode === 'table' && (
              <div className="w-full h-full min-h-0 overflow-hidden flex flex-col">
                <ThermalMetricsMatrix
                  reconstruction={reconstruction}
                  maxHeight="100%"
                />
              </div>
            )}

            {viewMode === 'transect' && (
              <div className="w-full h-full min-h-0 overflow-hidden flex flex-col">
                <VerticalTransectPlot
                  transect={transectData}
                  loading={transectLoading}
                  onSelectPreset={handleSelectPresetTransect}
                  onClearTransect={handleClearTransect}
                />
              </div>
            )}
          </div>

          {/* Dialog Footer with explicit Close Button */}
          <div className="flex items-center justify-end pt-3 border-t border-[#e2e8f0] flex-shrink-0">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsProfileExpanded(false)}
              className="h-8 px-4 text-xs font-medium rounded-lg active:scale-95 transition-all cursor-pointer"
            >
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

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
              <span className="text-[13px] text-[#64748d] font-mono mt-0.5 block">
                Surface observations → latent representation → subsurface reconstruction → validation
              </span>
            </div>

            <TabsList className="h-8.5 p-0.5 rounded-lg bg-[#f1f5f9] border border-[#e2e8f0]">
              <TabsTrigger value="drivers" className="text-xs px-3 rounded-md data-[state=active]:bg-white data-[state=active]:text-[#0d253d] data-[state=active]:shadow-xs">
                Surface Observations (7)
              </TabsTrigger>

              <TabsTrigger value="departure" className="text-xs px-3 rounded-md data-[state=active]:bg-white data-[state=active]:text-[#0d253d] data-[state=active]:shadow-xs">
                Reconstruction Departure
              </TabsTrigger>

              <TabsTrigger value="skill" className="text-xs px-3 rounded-md data-[state=active]:bg-white data-[state=active]:text-[#0d253d] data-[state=active]:shadow-xs">
                Depth-wise Model Skill
              </TabsTrigger>

              <TabsTrigger value="manifold" className="text-xs px-3 rounded-md data-[state=active]:bg-white data-[state=active]:text-[#0d253d] data-[state=active]:shadow-xs">
                128-D Latent Manifold
              </TabsTrigger>

              <TabsTrigger value="validation" className="text-xs px-3 rounded-md data-[state=active]:bg-white data-[state=active]:text-[#0d253d] data-[state=active]:shadow-xs">
                In-Situ Argo Validation
              </TabsTrigger>

              <TabsTrigger value="provenance" className="text-xs px-3 rounded-md data-[state=active]:bg-white data-[state=active]:text-[#0d253d] data-[state=active]:shadow-xs">
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

          <TabsContent value="departure" className="pt-4 m-0">
            <ReconstructionDeparturePanel
              date={effectiveDate}
              latitude={latitude}
              longitude={longitude}
              isMock={reconstruction?.is_mock ?? false}
              onSelectDate={onSelectDate}
            />
          </TabsContent>

          <TabsContent value="skill" className="pt-4 m-0">
            <DepthWiseSkillProfilePanel
              date={effectiveDate}
              isMock={reconstruction?.is_mock ?? false}
              onSelectDate={onSelectDate}
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
