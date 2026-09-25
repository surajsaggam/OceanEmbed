import React from 'react';
import type { ReconstructionResponse } from '@/types/api';

interface ScientificProvenanceCardProps {
  reconstruction: ReconstructionResponse | null;
}

export const ScientificProvenanceCard: React.FC<ScientificProvenanceCardProps> = ({
  reconstruction,
}) => {
  if (!reconstruction) {
    return (
      <div className="p-8 text-center text-[#64748d] text-sm border border-[#e3e8ee] rounded-xl bg-white shadow-xs">
        No active reconstruction. Execute reconstruction to inspect scientific data lineage.
      </div>
    );
  }

  const { model, argo_comparison: argo } = reconstruction;

  const provenanceItems = [
    { label: 'Model Identifier & Version', value: `${model.name} ${model.version}` },
    { label: 'Neural Architecture', value: 'Dual-Path Multimodal Residual MLP (Spatial CNN + Pointwise MLP)' },
    { label: 'Feature Representation', value: '128-Dimensional Continuous Latent State (Z ∈ ℝ¹²⁸)' },
    { label: 'Profile Inference Latency', value: `${model.inference_time_ms} ms (Single column vertical profile)` },
    { label: 'Training Reanalysis Target', value: 'GLORYS12V1 Reanalysis (CMEMS, 1993–2020)' },
    { label: 'Spatial Discretization', value: '0.083° resampled to 0.25° × 0.25° regular grid' },
    { label: 'Vertical Discretization', value: '15 Standard Oceanographic Depths (0 to 1000 m)' },
    {
      label: 'Independent Validation Source',
      value: argo ? `In-situ Argo Float (${argo.float_id}) · Distance: ${argo.distance_km.toFixed(1)} km` : 'Strictly held out from training (INCOIS Argo LAS)',
    },
    { label: 'Execution Trace ID', value: reconstruction.request_id },
    { label: 'Provider Runtime', value: model.provider_type },
  ];

  return (
    <div className="space-y-4" role="region" aria-label="Scientific Provenance & Audit">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#e2e8f0] pb-2">
        <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
          Scientific Provenance & Model Audit
        </span>
        <span className="text-[13px] font-mono text-[#64748d]">
          {reconstruction.is_mock ? 'Mode: Synthetic Climatology' : 'Mode: Production Inference Checkpoint'}
        </span>
      </div>

      {/* Structured Specification Table */}
      <div className="rounded-xl border border-[#e3e8ee] bg-white overflow-hidden shadow-xs">
        <div className="divide-y divide-[#f1f5f9] text-sm font-mono">
          {provenanceItems.map((item) => (
            <div key={item.label} className="grid grid-cols-1 sm:grid-cols-12 px-4 py-2.5 hover:bg-[#f8fafc] transition-colors">
              <span className="sm:col-span-4 font-sans font-medium text-[#64748d]">
                {item.label}
              </span>
              <span className="sm:col-span-8 text-[#0d253d] font-medium break-all">
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Operational Guardrail Note */}
      <div className="px-4 py-3 rounded-xl bg-[#f8fafc] border border-[#e2e8f0] text-sm text-[#64748d] leading-relaxed">
        <span className="font-semibold text-[#0d253d] block mb-0.5">Scientific Limitation & Mandate</span>
        OceanEmbed reconstructs subsurface ocean temperature over the North Indian Ocean from daily multi-source satellite surface observations. It complements sparse observing arrays but does not replace moorings, research vessels, or in-situ Argo profiling floats.
      </div>
    </div>
  );
};

