import * as React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Separator } from '@/components/ui/separator';
import { NIO_DOMAIN, STANDARD_DEPTHS_M } from '@/lib/ocean';
import oceanIqLogo from '@/assets/oceaniq-logo.png';

interface AboutDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const AboutDialog: React.FC<AboutDialogProps> = ({
  open,
  onOpenChange,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto bg-white border-[#e3e8ee] text-[#0d253d] p-6 shadow-xl">
        <DialogHeader>
          <div className="flex items-center gap-2.5 mb-1.5">
            <img
              src={oceanIqLogo}
              alt="OceanIQ Logo"
              className="size-7 object-contain select-none"
            />
            <span className="font-light text-xl tracking-[-0.02em] font-sans select-none flex items-center">
              <span className="text-[#0d253d]">Ocean</span>
              <span className="font-medium bg-gradient-to-r from-[#00b4f0] via-[#006ce6] to-[#004dc7] bg-clip-text text-transparent ml-0.5">
                IQ
              </span>
            </span>
          </div>
          <DialogTitle className="text-lg font-medium tracking-tight text-[#0d253d]">
            About OceanIQ
          </DialogTitle>
          <DialogDescription className="text-sm text-[#64748d]">
            Reading the depths through the surface · Subsurface ocean thermal reconstruction
          </DialogDescription>
        </DialogHeader>

        <Separator className="bg-[#e3e8ee] my-2" />

        <div className="space-y-4 text-sm leading-relaxed text-[#273951]">
          {/* Mission & Purpose */}
          <div className="border border-[#e3e8ee] rounded-xl p-4 space-y-2 bg-white">
            <div className="font-semibold text-xs text-[#0d253d] uppercase tracking-wider">
              What OceanIQ Does
            </div>
            <p className="text-[#475569]">
              <b>OceanIQ</b> reconstructs vertical ocean temperature profiles from the sea surface down to <b>1000 meters depth</b> across the <b>North Indian Ocean</b> basin ({NIO_DOMAIN.latMin}°N–{NIO_DOMAIN.latMax}°N, {NIO_DOMAIN.lonMin}°E–{NIO_DOMAIN.lonMax}°E) using daily multi-source satellite observations.
            </p>
          </div>

          {/* The Scientific Problem */}
          <div className="border border-[#e3e8ee] rounded-xl p-4 space-y-2 bg-white">
            <div className="font-semibold text-xs text-[#0d253d] uppercase tracking-wider">
              The Physical Challenge
            </div>
            <p className="text-[#475569]">
              Satellite radiometers and altimeters measure only the ocean’s skin layer (surface temperature, salinity, sea height, currents, and winds). However, the majority of ocean heat content, thermal inertia, and cyclone-fueling energy resides hidden tens to hundreds of meters below the surface in the thermocline.
            </p>
            <p className="text-[#475569]">
              Traditional subsurface measurements rely on sparse autonomous Argo floats and moorings. OceanIQ bridges this spatial and temporal gap by learning the physical coupling between surface signatures and 3D subsurface thermal structure.
            </p>
          </div>

          {/* Key Capabilities */}
          <div className="border border-[#e3e8ee] rounded-xl p-4 space-y-2 bg-white">
            <div className="font-semibold text-xs text-[#0d253d] uppercase tracking-wider">
              Key Capabilities & Output
            </div>
            <ul className="list-disc list-inside space-y-1.5 pl-1 text-[#475569]">
              <li>
                <b>15 Standard Oceanographic Depths:</b> Continuous vertical temperature soundings at {STANDARD_DEPTHS_M.join(', ')} meters.
              </li>
              <li>
                <b>Tropical Cyclone Heat Potential:</b> Instant derivation of the <b>D26</b> isotherm depth (26°C isotherm threshold) critical for Bay of Bengal and Arabian Sea cyclone intensification analysis.
              </li>
              <li>
                <b>Mixed Layer Depth (MLD):</b> Automated extraction of the surface mixing boundary layer where temperature drops 0.2°C from surface values.
              </li>
              <li>
                <b>Blind In-Situ Argo Float Validation:</b> Collocation and comparative evaluation against independent in-situ Argo profiles.
              </li>
            </ul>
          </div>

          {/* Scientific Guardrail */}
          <div className="p-3.5 rounded-xl bg-[#f8fafc] border border-[#e2e8f0] text-xs text-[#64748d]">
            <b className="text-[#0d253d]">Scientific Integrity Notice:</b> OceanIQ is an estimation framework designed to complement physical oceanographic observation systems. It does not replace direct in-situ Argo profiling floats, moored buoys, or CTD research voyages.
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
