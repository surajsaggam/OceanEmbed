import React from 'react';
import type { HealthResponse, ReconstructionResponse } from '@/types/api';

interface AppHeaderProps {
  health?: HealthResponse | null;
  reconstruction?: ReconstructionResponse | null;
}

export const AppHeader: React.FC<AppHeaderProps> = () => {
  return (
    <header
      className="w-full bg-white border-b border-[#e3e8ee] px-6 lg:px-8 py-3.5 flex items-center gap-3"
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
    </header>
  );
};

