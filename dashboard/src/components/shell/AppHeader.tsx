import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ModelSpecsDialog } from '../dialogs/ModelSpecsDialog';
import type { HealthResponse, ReconstructionResponse } from '@/types/api';

interface AppHeaderProps {
  health: HealthResponse | null;
  reconstruction: ReconstructionResponse | null;
}

export const AppHeader: React.FC<AppHeaderProps> = ({ health, reconstruction }) => {
  const [specsOpen, setSpecsOpen] = useState(false);
  const isHealthy = health?.status === 'healthy';
  const isMock = reconstruction ? reconstruction.is_mock : (health?.is_mock ?? true);
  const latencyMs = reconstruction?.model.inference_time_ms;

  return (
    <>
      <header
        className="w-full bg-white/95 backdrop-blur-md border-b border-[#e3e8ee] px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30"
        role="banner"
      >
        {/* Brand & Scientific Context */}
        <div className="flex items-center gap-3 shrink-0">
          <span className="font-light text-2xl text-[#0d253d] tracking-[-0.02em] font-sans select-none">
            OceanEmbed
          </span>
          <span className="h-4 w-px bg-[#e3e8ee] hidden sm:inline-block" aria-hidden="true" />
          <span className="text-sm text-[#64748d] font-normal hidden sm:inline-block">
            Subsurface Ocean Thermal Reconstruction · North Indian Ocean (5°–30°N, 45°–105°E)
          </span>
        </div>

        {/* Global Telemetry & Science Specs Trigger */}
        <div className="flex items-center gap-4 shrink-0 text-sm">
          {/* API Health & Latency */}
          <div className="flex items-center gap-2 text-[#273951]">
            <span
              className={`size-2 rounded-full ${
                isHealthy ? 'bg-emerald-500' : 'bg-amber-500'
              }`}
              aria-hidden="true"
            />
            <span className="text-[13px] font-mono uppercase tracking-wider text-[#64748d]">
              {health ? health.status : 'Connecting'}
            </span>
            {latencyMs !== undefined && (
              <span className="font-mono text-[#0d253d] font-medium tabular-nums text-[13px]">
                · {latencyMs} ms
              </span>
            )}
          </div>

          <span className="h-3.5 w-px bg-[#e3e8ee] hidden md:inline-block" aria-hidden="true" />

          {/* Provenance Indicator */}
          <span className="text-[13px] font-mono text-[#64748d] hidden md:inline-block">
            {isMock ? 'Data: Synthetic Climatology' : 'Data: INCOIS Observation Benchmark'}
          </span>

          {/* Science & Specs Dialog Trigger */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSpecsOpen(true)}
            className="text-sm text-[#273951] hover:text-[#0d253d] border-[#e3e8ee] h-8 px-3.5"
            title="Inspect dual-path neural architecture and scientific formulation"
          >
            Methodology & Architecture
          </Button>
        </div>
      </header>

      <ModelSpecsDialog open={specsOpen} onOpenChange={setSpecsOpen} />
    </>
  );
};

