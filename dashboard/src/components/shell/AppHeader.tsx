import React, { useState } from 'react';
import oceanIqLogo from '@/assets/oceaniq-logo.png';
import { AboutDialog } from '../dialogs/AboutDialog';
import { ModelSpecsDialog } from '../dialogs/ModelSpecsDialog';
import type { HealthResponse, ReconstructionResponse } from '@/types/api';

interface AppHeaderProps {
  health?: HealthResponse | null;
  reconstruction?: ReconstructionResponse | null;
}

export const AppHeader: React.FC<AppHeaderProps> = () => {
  const [aboutOpen, setAboutOpen] = useState(false);
  const [methodologyOpen, setMethodologyOpen] = useState(false);

  return (
    <>
      <header
        className="w-full bg-white border-b border-[#e3e8ee] px-6 lg:px-8 py-3.5 flex items-center justify-between gap-4"
        role="banner"
      >
        {/* Brand & Scientific Context */}
        <div className="flex items-center gap-3 shrink-0">
          <img
            src={oceanIqLogo}
            alt="OceanIQ Logo"
            className="size-8.5 object-contain select-none"
          />
          <span className="font-light text-2xl tracking-[-0.02em] font-sans select-none flex items-center">
            <span className="text-[#0d253d]">Ocean</span>
            <span className="font-medium bg-gradient-to-r from-[#00b4f0] via-[#006ce6] to-[#004dc7] bg-clip-text text-transparent ml-0.5">
              IQ
            </span>
          </span>
          <span className="h-4 w-px bg-[#e3e8ee] hidden sm:inline-block" aria-hidden="true" />
          <span className="text-sm text-[#64748d] font-normal hidden sm:inline-block">
            Reading the depths through the surface.
          </span>
        </div>

        {/* Subtle Context Links (About · Methodology) */}
        <nav aria-label="Platform Context" className="flex items-center gap-1.5 sm:gap-2 text-sm shrink-0">
          <button
            type="button"
            onClick={() => setAboutOpen(true)}
            className="text-xs sm:text-sm font-medium text-[#475569] hover:text-[#0d253d] hover:bg-[#f1f5f9] transition-all cursor-pointer tracking-[-0.01em] py-1.5 px-3 rounded-lg active:scale-95 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-[#533afd]/20"
          >
            About
          </button>
          <span className="text-[#cbd5e1] select-none text-xs" aria-hidden="true">
            ·
          </span>
          <button
            type="button"
            onClick={() => setMethodologyOpen(true)}
            className="text-xs sm:text-sm font-medium text-[#475569] hover:text-[#0d253d] hover:bg-[#f1f5f9] transition-all cursor-pointer tracking-[-0.01em] py-1.5 px-3 rounded-lg active:scale-95 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-[#533afd]/20"
          >
            Methodology
          </button>
        </nav>
      </header>

      {/* Modals */}
      <AboutDialog open={aboutOpen} onOpenChange={setAboutOpen} />
      <ModelSpecsDialog open={methodologyOpen} onOpenChange={setMethodologyOpen} />
    </>
  );
};

