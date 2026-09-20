import React from 'react';
import type { ArgoObservation } from '@/types/api';

interface ArgoValidationPanelProps {
  argo: ArgoObservation | null | undefined;
}

export const ArgoValidationPanel: React.FC<ArgoValidationPanelProps> = ({ argo }) => {
  if (!argo) {
    return (
      <div className="p-8 text-center text-[#64748d] text-sm border border-[#e3e8ee] rounded-lg bg-white space-y-2">
        <div className="font-semibold text-base text-[#0d253d]">
          No Collocated In-Situ Argo Float in Validation Window
        </div>
        <p className="text-[#64748d] max-w-lg mx-auto leading-relaxed">
          There is no quality-controlled in-situ Argo float observation matching this coordinate and date neighborhood (±0.25°, ±24h) in the local catalog. Ground-truth Argo float profiles from the INCOIS Live Access Server (LAS) are held out strictly for independent blind validation.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4" role="region" aria-label="In-Situ Argo Float Validation Report">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#e2e8f0] pb-2">
        <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
          Independent In-Situ Float Validation (WMO: {argo.float_id})
        </span>
        <span className="text-[13px] font-mono text-[#64748d]">
          {argo.is_mock ? 'Validation Benchmark: Synthetic Demo Float' : 'Validation Benchmark: INCOIS Argo LAS In-Situ'}
        </span>
      </div>

      {/* Two-Column Scientific Summary Sheet */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Float Metadata Table */}
        <div className="rounded-lg border border-[#e3e8ee] bg-white overflow-hidden">
          <div className="px-4 py-2.5 bg-[#f8fafc] border-b border-[#e2e8f0] text-sm font-medium text-[#64748d]">
            In-Situ Float Cycle Metadata
          </div>
          <div className="p-4 space-y-2 text-sm font-mono">
            <div className="flex justify-between py-1 border-b border-[#f1f5f9]">
              <span className="text-[#64748d] font-sans">WMO Float Identifier</span>
              <span className="text-[#0d253d] font-semibold">{argo.float_id}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#f1f5f9]">
              <span className="text-[#64748d] font-sans">Profile Observation Date</span>
              <span className="text-[#0d253d]">{argo.date}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#f1f5f9]">
              <span className="text-[#64748d] font-sans">Float Coordinates</span>
              <span className="text-[#0d253d]">
                {argo.latitude.toFixed(2)}°N, {argo.longitude.toFixed(2)}°E
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#64748d] font-sans">Radial Distance to Station</span>
              <span className="text-[#0d253d] tabular-nums">{argo.distance_km.toFixed(1)} km</span>
            </div>
          </div>
        </div>

        {/* Quantitative Residual Metrics Table */}
        <div className="rounded-lg border border-[#e3e8ee] bg-white overflow-hidden">
          <div className="px-4 py-2.5 bg-[#f8fafc] border-b border-[#e2e8f0] text-sm font-medium text-[#64748d]">
            Vertical Reconstruction Error Metrics (0–1000m)
          </div>
          <div className="p-4 space-y-2 text-sm font-mono">
            <div className="flex justify-between py-1 border-b border-[#f1f5f9]">
              <span className="text-[#64748d] font-sans">Root Mean Square Error (RMSE)</span>
              <span className="text-[#e11d48] font-bold tabular-nums">
                {argo.rmse !== null && argo.rmse !== undefined ? `${argo.rmse.toFixed(2)} °C` : '—'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#f1f5f9]">
              <span className="text-[#64748d] font-sans">Mean Absolute Error (MAE)</span>
              <span className="text-[#b45309] font-bold tabular-nums">
                {argo.mae !== null && argo.mae !== undefined ? `${argo.mae.toFixed(2)} °C` : '—'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#f1f5f9]">
              <span className="text-[#64748d] font-sans">Mean Residual Bias</span>
              <span className="text-[#0284c7] font-bold tabular-nums">
                {argo.bias !== null && argo.bias !== undefined
                  ? `${argo.bias > 0 ? '+' : ''}${argo.bias.toFixed(2)} °C`
                  : '—'}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#64748d] font-sans">Validated Standard Levels</span>
              <span className="text-[#0d253d]">15 Standard Depths (0–1000m)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Benchmark Guardrail Note */}
      <div className="px-4 py-3 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] text-sm text-[#64748d] leading-relaxed">
        <span className="font-semibold text-[#0d253d] block mb-0.5">Independent Validation Protocol</span>
        Quality-controlled Argo CTD observations from the Indian National Centre for Ocean Information Services (INCOIS) Live Access Server are kept strictly independent from training datasets and evaluated as blind reference profiles.
      </div>
    </div>
  );
};

