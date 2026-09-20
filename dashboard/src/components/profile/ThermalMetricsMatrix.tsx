import React from 'react';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import type { ReconstructionResponse } from '@/types/api';

interface ThermalMetricsMatrixProps {
  reconstruction: ReconstructionResponse | null;
  maxHeight?: string;
}

export const ThermalMetricsMatrix: React.FC<ThermalMetricsMatrixProps> = ({
  reconstruction,
  maxHeight = '460px',
}) => {
  if (!reconstruction) {
    return (
      <div className="h-full flex items-center justify-center text-[#64748d] text-sm p-8 text-center border border-[#e3e8ee] rounded-xl bg-white shadow-xs">
        No reconstruction data available. Specify location/date and click Reconstruct.
      </div>
    );
  }

  const argo = reconstruction.argo_comparison;

  return (
    <div
      className="w-full rounded-xl border border-[#e3e8ee] bg-white overflow-y-auto shadow-xs"
      style={{ maxHeight }}
    >
      <Table aria-label="15 Standard Depths Thermal Sounding Matrix">
        <TableHeader className="sticky top-0 bg-[#f8fafc] border-b border-[#e2e8f0] z-10">
          <TableRow>
            <TableHead className="font-sans text-sm text-[#64748d]">Depth</TableHead>
            <TableHead className="font-sans text-sm text-right text-[#64748d]">Recon (°C)</TableHead>
            {argo && <TableHead className="font-sans text-sm text-right text-[#64748d]">Argo (°C)</TableHead>}
            {argo && <TableHead className="font-sans text-sm text-right text-[#64748d]">ΔT (°C)</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody className="font-mono text-sm">
          {reconstruction.depths_m.map((depth, idx) => {
            const recT = reconstruction.temperature_c[idx];
            const argoT = argo ? argo.temperature_c[idx] : null;
            const delta = argoT !== null && argoT !== undefined ? recT - argoT : null;

            return (
              <TableRow key={depth} className="hover:bg-[#f8fafc] border-b border-[#f1f5f9]">
                <TableCell className="font-sans font-medium text-[#0d253d]">
                  {depth} m
                </TableCell>
                <TableCell className="text-right text-[#3b49df] font-semibold tabular-nums">
                  {recT.toFixed(2)}
                </TableCell>
                {argo && (
                  <TableCell className="text-right text-[#e11d48] font-medium tabular-nums">
                    {argoT !== null ? argoT.toFixed(2) : '—'}
                  </TableCell>
                )}
                {argo && (
                  <TableCell className="text-right font-mono tabular-nums text-sm">
                    {delta !== null ? (
                      <span
                        className={`font-medium ${
                          Math.abs(delta) < 0.05
                            ? 'text-[#64748d]'
                            : delta > 0
                            ? 'text-[#b45309]'
                            : 'text-[#0284c7]'
                        }`}
                      >
                        {delta > 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-[#94a3b8]">—</span>
                    )}
                  </TableCell>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
};
