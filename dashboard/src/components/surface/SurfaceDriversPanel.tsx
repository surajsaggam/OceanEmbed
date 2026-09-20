import React from 'react';
import { Table } from '@heroui/react';
import { calculateVectorKinematics } from '@/lib/ocean';
import type { SurfaceContext } from '@/types/api';

interface SurfaceDriversPanelProps {
  surface: SurfaceContext | null;
  isMock: boolean;
}

export const SurfaceDriversPanel: React.FC<SurfaceDriversPanelProps> = ({
  surface,
  isMock,
}) => {
  if (!surface) {
    return (
      <div className="p-8 text-center text-[#64748d] text-sm border border-[#e3e8ee] rounded-xl bg-white">
        Trigger reconstruction to inspect the 7 multi-source satellite boundary observations.
      </div>
    );
  }

  const currentKinematics = calculateVectorKinematics(
    surface.current_u_ms,
    surface.current_v_ms
  );

  const windKinematics = calculateVectorKinematics(
    surface.wind_u_ms,
    surface.wind_v_ms
  );

  const observations = [
    {
      name: 'Sea Surface Temperature',
      symbol: 'SST (Ts)',
      value: surface.sst_c.toFixed(2),
      unit: '°C',
      source: 'OSTIA MW+IR (0.05° Copernicus)',
      role: 'Upper boundary Dirichlet condition; governs ocean skin heating and mixed layer base.',
    },
    {
      name: 'Sea Surface Salinity',
      symbol: 'SSS (Ss)',
      value: surface.sss_psu.toFixed(2),
      unit: 'PSU',
      source: 'SMAP / SMOS L3 (NASA JPL / ESA)',
      role: 'Haline buoyancy control; resolves barrier layer thickness and freshwater plumes in the Bay of Bengal.',
    },
    {
      name: 'Sea Surface Height Anomaly',
      symbol: 'SSHA (η)',
      value: `${surface.ssh_m >= 0 ? '+' : ''}${surface.ssh_m.toFixed(2)}`,
      unit: 'm',
      source: 'DUACS Multi-Mission Altimetry (CMEMS)',
      role: 'Integrated baroclinic proxy; directly relates to steric height and pycnocline/thermocline depth displacement.',
    },
    {
      name: 'Zonal Surface Current',
      symbol: 'u_curr',
      value: `${surface.current_u_ms >= 0 ? '+' : ''}${surface.current_u_ms.toFixed(2)}`,
      unit: 'm/s',
      source: 'OSCAR / Copernicus Ocean Currents',
      role: 'Zonal advective heat flux component across equatorial wave guides and boundary currents.',
    },
    {
      name: 'Meridional Surface Current',
      symbol: 'v_curr',
      value: `${surface.current_v_ms >= 0 ? '+' : ''}${surface.current_v_ms.toFixed(2)}`,
      unit: 'm/s',
      source: 'OSCAR / Copernicus Ocean Currents',
      role: 'Meridional advective heat flux component; tracks West India and East India coastal currents.',
    },
    {
      name: 'Zonal 10m Wind',
      symbol: 'u_wind',
      value: `${surface.wind_u_ms >= 0 ? '+' : ''}${surface.wind_u_ms.toFixed(2)}`,
      unit: 'm/s',
      source: 'ASCAT / MetOp Scatterometer',
      role: 'Zonal momentum flux driving surface turbulence, evaporation, and shear-induced vertical mixing.',
    },
    {
      name: 'Meridional 10m Wind',
      symbol: 'v_wind',
      value: `${surface.wind_v_ms >= 0 ? '+' : ''}${surface.wind_v_ms.toFixed(2)}`,
      unit: 'm/s',
      source: 'ASCAT / MetOp Scatterometer',
      role: 'Meridional wind stress curl driving coastal upwelling (e.g. Somali Current / SW monsoon jets).',
    },
  ];

  return (
    <div className="space-y-4" role="region" aria-label="7 Multi-Source Satellite Boundary Drivers">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#e2e8f0] pb-2">
        <span className="text-[13px] font-semibold text-[#0d253d] uppercase tracking-wider">
          Multi-Source Satellite Surface Observations (7 Channels)
        </span>
        <span className="text-[13px] font-mono text-[#64748d]">
          {isMock ? 'Source: Synthetic Climatological Proxy' : 'Source: Observed Satellite Blend'}
        </span>
      </div>

      {/* Publication-Quality Scientific Data Table */}
      <div className="rounded-xl border border-[#e3e8ee] bg-white overflow-hidden shadow-xs">
        <Table>
          <Table.ScrollContainer>
            <Table.Content aria-label="Multi-Source Satellite Surface Observations" className="w-full min-w-[700px] text-left text-sm border-collapse">
              <Table.Header>
                <Table.Column isRowHeader className="py-2.5 px-4 font-medium text-[#64748d] bg-[#f8fafc] border-b border-[#e2e8f0] text-left font-sans">
                  Parameter
                </Table.Column>
                <Table.Column className="py-2.5 px-3 font-medium text-[#64748d] bg-[#f8fafc] border-b border-[#e2e8f0] text-left font-sans">
                  Symbol
                </Table.Column>
                <Table.Column className="py-2.5 px-4 font-medium text-[#64748d] bg-[#f8fafc] border-b border-[#e2e8f0] text-right font-sans">
                  Observed Value
                </Table.Column>
                <Table.Column className="py-2.5 px-3 font-medium text-[#64748d] bg-[#f8fafc] border-b border-[#e2e8f0] text-left font-sans">
                  Unit
                </Table.Column>
                <Table.Column className="py-2.5 px-4 font-medium text-[#64748d] bg-[#f8fafc] border-b border-[#e2e8f0] text-left font-sans">
                  Sensor / Platform
                </Table.Column>
                <Table.Column className="py-2.5 px-4 font-medium text-[#64748d] bg-[#f8fafc] border-b border-[#e2e8f0] text-left font-sans">
                  Physical Role in Inversion
                </Table.Column>
              </Table.Header>
              <Table.Body className="divide-y divide-[#f1f5f9] text-[#273951]">
                {observations.map((obs) => (
                  <Table.Row key={obs.symbol} id={obs.symbol} className="hover:bg-[#f8fafc] transition-colors border-b border-[#f1f5f9]">
                    <Table.Cell className="py-2.5 px-4 font-medium text-[#0d253d] whitespace-nowrap">
                      {obs.name}
                    </Table.Cell>
                    <Table.Cell className="py-2.5 px-3 font-mono text-[13px] text-[#64748d] whitespace-nowrap">
                      {obs.symbol}
                    </Table.Cell>
                    <Table.Cell className="py-2.5 px-4 font-mono font-semibold text-[#0d253d] text-right tabular-nums whitespace-nowrap">
                      {obs.value}
                    </Table.Cell>
                    <Table.Cell className="py-2.5 px-3 text-[#64748d] whitespace-nowrap">
                      {obs.unit}
                    </Table.Cell>
                    <Table.Cell className="py-2.5 px-4 text-[#475569] whitespace-nowrap">
                      {obs.source}
                    </Table.Cell>
                    <Table.Cell className="py-2.5 px-4 text-[#64748d] text-[13px] leading-relaxed max-w-md">
                      {obs.role}
                    </Table.Cell>
                  </Table.Row>
                ))}
              </Table.Body>
            </Table.Content>
          </Table.ScrollContainer>
        </Table>
      </div>

      {/* Derived Kinematics Strip */}
      <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-2.5 rounded-xl bg-[#f8fafc] border border-[#e2e8f0] text-sm font-mono">
        <div className="flex items-center gap-2">
          <span className="text-[#64748d] font-sans font-medium">Surface Current Drift:</span>
          <span className="text-[#0d253d] font-semibold tabular-nums">
            {currentKinematics.speed.toFixed(2)} m/s
          </span>
          <span className="text-[#64748d]">at</span>
          <span className="text-[#0d253d] font-semibold tabular-nums">
            {currentKinematics.headingDeg.toFixed(0)}°
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[#64748d] font-sans font-medium">10m Surface Wind:</span>
          <span className="text-[#0d253d] font-semibold tabular-nums">
            {windKinematics.speed.toFixed(1)} m/s
          </span>
          <span className="text-[#64748d]">at</span>
          <span className="text-[#0d253d] font-semibold tabular-nums">
            {windKinematics.headingDeg.toFixed(0)}°
          </span>
        </div>
      </div>
    </div>
  );
};

