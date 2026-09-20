import React from 'react';
import { Table } from '@heroui/react';
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
      className="w-full h-full rounded-xl border border-[#e3e8ee] bg-white flex flex-col overflow-hidden shadow-xs"
      style={{ maxHeight }}
    >
      <Table className="w-full h-full flex flex-col overflow-hidden">
        <Table.ScrollContainer className="w-full flex-1 overflow-y-auto overflow-x-hidden [scrollbar-gutter:stable] pr-1.5">
          <Table.Content aria-label="15 Standard Depths Thermal Sounding Matrix" className="w-full min-w-full text-left text-sm border-collapse">
            <Table.Header className="sticky top-0 bg-[#f8fafc] border-b border-[#e2e8f0] z-10">
              <Table.Column isRowHeader className="py-2.5 px-3 sm:px-4 font-sans text-xs sm:text-sm text-[#64748d] text-left font-medium whitespace-nowrap">
                Depth
              </Table.Column>
              <Table.Column className={`py-2.5 pl-3 sm:pl-4 ${argo ? 'pr-3 sm:pr-4' : 'pr-6 sm:pr-8'} font-sans text-xs sm:text-sm text-[#64748d] text-right font-medium whitespace-nowrap`}>
                Recon (°C)
              </Table.Column>
              {argo && (
                <Table.Column className="py-2.5 px-3 sm:px-4 font-sans text-xs sm:text-sm text-[#64748d] text-right font-medium whitespace-nowrap">
                  Argo (°C)
                </Table.Column>
              )}
              {argo && (
                <Table.Column className="py-2.5 pl-3 sm:pl-4 pr-6 sm:pr-8 font-sans text-xs sm:text-sm text-[#64748d] text-right font-medium whitespace-nowrap">
                  ΔT (°C)
                </Table.Column>
              )}
            </Table.Header>
            <Table.Body className="font-mono text-xs sm:text-sm divide-y divide-[#f1f5f9]">
              {reconstruction.depths_m.map((depth, idx) => {
                const recT = reconstruction.temperature_c[idx];
                const argoT = argo ? argo.temperature_c[idx] : null;
                const delta = argoT !== null && argoT !== undefined ? recT - argoT : null;

                return (
                  <Table.Row key={depth} id={String(depth)} className="hover:bg-[#f8fafc] border-b border-[#f1f5f9] transition-colors">
                    <Table.Cell className="py-2 px-3 sm:px-4 font-sans font-medium text-[#0d253d] whitespace-nowrap">
                      {depth} m
                    </Table.Cell>
                    <Table.Cell className={`py-2 pl-3 sm:pl-4 ${argo ? 'pr-3 sm:pr-4' : 'pr-6 sm:pr-8'} text-right text-[#3b49df] font-semibold tabular-nums whitespace-nowrap`}>
                      {recT.toFixed(2)}
                    </Table.Cell>
                    {argo && (
                      <Table.Cell className="py-2 px-3 sm:px-4 text-right text-[#e11d48] font-medium tabular-nums whitespace-nowrap">
                        {argoT !== null ? argoT.toFixed(2) : '—'}
                      </Table.Cell>
                    )}
                    {argo && (
                      <Table.Cell className="py-2 pl-3 sm:pl-4 pr-6 sm:pr-8 text-right font-mono tabular-nums text-xs sm:text-sm whitespace-nowrap">
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
                      </Table.Cell>
                    )}
                  </Table.Row>
                );
              })}
            </Table.Body>
          </Table.Content>
        </Table.ScrollContainer>
      </Table>
    </div>
  );
};
