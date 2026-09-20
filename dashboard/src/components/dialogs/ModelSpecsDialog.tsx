import * as React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Separator } from '@/components/ui/separator';
import { STANDARD_DEPTHS_M, NIO_DOMAIN } from '@/lib/ocean';

interface ModelSpecsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const ModelSpecsDialog: React.FC<ModelSpecsDialogProps> = ({
  open,
  onOpenChange,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto bg-white border-[#e3e8ee] text-[#0d253d] p-6 shadow-xl">
        <DialogHeader>
          <div className="text-[11px] font-mono text-[#64748d] uppercase tracking-wider mb-1">
            Institutional Research Specification · MoES / INCOIS
          </div>
          <DialogTitle className="text-lg font-medium tracking-tight text-[#0d253d]">
            OceanEmbed Architecture & Scientific Methodology
          </DialogTitle>
          <DialogDescription className="text-xs text-[#64748d]">
            Satellite-Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature
          </DialogDescription>
        </DialogHeader>

        <Separator className="bg-[#e3e8ee] my-2" />

        <div className="space-y-4 text-xs leading-relaxed text-[#273951]">
          {/* Domain & Depths */}
          <div className="border border-[#e3e8ee] rounded-lg p-3.5 space-y-2 bg-white">
            <div className="font-semibold text-xs text-[#0d253d] uppercase tracking-wider">
              Geographic Domain & Standard Output Depths
            </div>
            <p className="text-[#475569]">
              Covers the <b>North Indian Ocean</b> basin (
              <span className="font-mono text-[#0d253d]">
                {NIO_DOMAIN.latMin}°N–{NIO_DOMAIN.latMax}°N, {NIO_DOMAIN.lonMin}°E–{NIO_DOMAIN.lonMax}°E
              </span>
              ) on a continuous daily <span className="font-mono text-[#0d253d]">{NIO_DOMAIN.gridResolution}° × {NIO_DOMAIN.gridResolution}° regular grid</span>.
              Reconstructs vertical thermal profiles across <b>15 standard oceanographic depths</b>:
            </p>
            <div className="flex flex-wrap gap-1.5 pt-1 font-mono text-[11px]">
              {STANDARD_DEPTHS_M.map((depth) => (
                <span
                  key={depth}
                  className="px-2 py-0.5 rounded bg-[#f8fafc] text-[#0d253d] border border-[#e2e8f0]"
                >
                  {depth}m
                </span>
              ))}
            </div>
          </div>
          {/* Neural Architecture */}
          <div className="border border-[#e3e8ee] rounded-lg p-3.5 space-y-2 bg-white">
            <div className="font-semibold text-xs text-[#0d253d] uppercase tracking-wider">
              Dual-Path Neural Architecture & 128-D Latent Bottleneck
            </div>
            <p className="text-[#475569]">
              Ingests a <b>14-channel input tensor</b> (7 multi-source physical surface variables + 7 binary observational validity masks):
            </p>
            <ul className="list-disc list-inside space-y-1 pl-1 text-[#475569]">
              <li>
                <b>Path A (Spatial CNN):</b> Parallel 3×3 and dilated 3×3 receptive fields capturing mesoscale thermal fronts and eddies.
              </li>
              <li>
                <b>Path B (Pointwise MLP):</b> 1×1 convolutional projections preserving direct vertical air-sea exchange without spatial smearing.
              </li>
              <li>
                <b>Feature Bottleneck:</b> Projection into an explicit latent ocean embedding <span className="font-mono text-[#533afd] font-semibold">Z ∈ ℝ¹²⁸</span>.
              </li>
              <li>
                <b>Attention Decoder:</b> Translates latent embeddings with residual skips into vertical temperatures at the 15 standard depths.
              </li>
            </ul>
          </div>

          {/* 7 Satellite Data Channels */}
          <div className="border border-[#e3e8ee] rounded-lg p-3.5 space-y-2 bg-white">
            <div className="font-semibold text-xs text-[#0d253d] uppercase tracking-wider">
              Multi-Source Satellite Boundary Drivers (7 Channels)
            </div>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1">
              <div className="p-2.5 bg-[#f8fafc] rounded border border-[#e2e8f0]">
                <span className="text-[#64748d] block text-[10px] uppercase font-sans">Sea Surface Temp (SST)</span>
                <span className="text-[#0d253d] font-semibold">OSTIA (0.05°) · CMEMS</span>
              </div>
              <div className="p-2.5 bg-[#f8fafc] rounded border border-[#e2e8f0]">
                <span className="text-[#64748d] block text-[10px] uppercase font-sans">Sea Surface Salinity (SSS)</span>
                <span className="text-[#0d253d] font-semibold">SMAP / SMOS L4 · NASA JPL</span>
              </div>
              <div className="p-2.5 bg-[#f8fafc] rounded border border-[#e2e8f0]">
                <span className="text-[#64748d] block text-[10px] uppercase font-sans">Sea Surface Height (SSH/SLA)</span>
                <span className="text-[#0d253d] font-semibold">DUACS Altimetry · CMEMS</span>
              </div>
              <div className="p-2.5 bg-[#f8fafc] rounded border border-[#e2e8f0]">
                <span className="text-[#64748d] block text-[10px] uppercase font-sans">Surface Currents (U, V)</span>
                <span className="text-[#0d253d] font-semibold">OSCAR L4 OC (0.25°) · PO.DAAC</span>
              </div>
            </div>
            <div className="p-2.5 bg-[#f8fafc] rounded border border-[#e2e8f0] font-mono text-[11px]">
              <span className="text-[#64748d] block text-[10px] uppercase font-sans">10m Surface Winds (U, V)</span>
              <span className="text-[#0d253d] font-semibold">CCMP v3.1 / ASCAT-C (0.25°) · PO.DAAC / EUMETSAT</span>
            </div>
          </div>

          {/* Validation Protocol */}
          <div className="border border-[#e3e8ee] rounded-lg p-3.5 space-y-2 bg-white">
            <div className="font-semibold text-xs text-[#0d253d] uppercase tracking-wider">
              Validation Protocol & Integrity Guardrails
            </div>
            <ul className="list-disc list-inside space-y-1 pl-1 text-[#475569]">
              <li>
                <b>Training Baseline:</b> GLORYS12V1 reanalysis (0.083° resampled to 0.25°). Acknowledged as a data-assimilative physical model, not absolute ground truth.
              </li>
              <li>
                <b>Independent Blind Validation:</b> Quality-controlled in-situ Argo float profiles from INCOIS LAS. Held out strictly; never used in training.
              </li>
              <li>
                <b>Disaster Management Utility:</b> Calculation of <b>D26</b> (depth of 26°C isotherm) and Tropical Cyclone Heat Potential (TCHP) in the Bay of Bengal & Arabian Sea.
              </li>
            </ul>
          </div>

          {/* Guardrail Warning */}
          <div className="p-3 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] text-xs text-[#64748d]">
            <b className="text-[#0d253d]">Scientific Guardrail:</b> OceanEmbed reconstructs subsurface thermal structure from multi-source satellite surface observations. It complements sparse in-situ arrays but does not replace Argo floats, moorings, or research vessels.
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

