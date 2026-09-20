import React, { useState, useMemo } from 'react';
import { ChevronDown, Check, CalendarIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import { Calendar } from '@/components/ui/calendar';
import { Popover, PopoverTrigger, PopoverContent } from '@/components/ui/popover';
import { OCEAN_PRESETS } from '@/data/presets';
import { isWithinNioDomain, OBSERVATION_DATE_RANGE } from '@/lib/ocean';
import type { OceanPreset } from '@/types/api';

interface TargetingStationBarProps {
  date: string;
  latitude: number;
  longitude: number;
  loading: boolean;
  onDateChange: (d: string) => void;
  onLatitudeChange: (lat: number) => void;
  onLongitudeChange: (lon: number) => void;
  onReconstruct: () => void;
  onSelectPreset: (preset: OceanPreset) => void;
}

export const TargetingStationBar: React.FC<TargetingStationBarProps> = ({
  date,
  latitude,
  longitude,
  loading,
  onDateChange,
  onLatitudeChange,
  onLongitudeChange,
  onReconstruct,
  onSelectPreset,
}) => {
  const [presetsOpen, setPresetsOpen] = useState(false);
  const [datePickerOpen, setDatePickerOpen] = useState(false);
  const isValid = isWithinNioDomain(latitude, longitude);

  const selectedDate = useMemo(() => {
    if (!date) return undefined;
    const [y, m, d] = date.split('-').map(Number);
    if (!y || !m || !d) return undefined;
    return new Date(y, m - 1, d);
  }, [date]);

  const handleSelectDate = (newDate: Date | undefined) => {
    if (newDate) {
      const year = newDate.getFullYear();
      const month = String(newDate.getMonth() + 1).padStart(2, '0');
      const day = String(newDate.getDate()).padStart(2, '0');
      onDateChange(`${year}-${month}-${day}`);
      setDatePickerOpen(false);
    }
  };

  const activePreset = OCEAN_PRESETS.find(
    (p) =>
      Math.abs(p.latitude - latitude) < 0.05 &&
      Math.abs(p.longitude - longitude) < 0.05
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && isValid && !loading) {
      onReconstruct();
    }
  };

  return (
    <div
      className="w-full bg-white border-b border-[#e3e8ee] px-6 lg:px-8 py-3.5 flex flex-wrap items-end justify-between gap-4"
      role="region"
      aria-label="Reconstruction Station Parameters"
    >
      {/* Parameter Inputs Group */}
      <div className="flex flex-wrap items-end gap-3 sm:gap-4">
        {/* Date Input */}
        <div>
          <label
            htmlFor="target-date-trigger"
            className="block text-[13px] font-medium text-[#64748d] mb-1 uppercase tracking-wider"
          >
            Observation Date
          </label>
          <Popover open={datePickerOpen} onOpenChange={setDatePickerOpen}>
            <PopoverTrigger asChild>
              <button
                id="target-date-trigger"
                type="button"
                disabled={loading}
                className="h-9 px-3 rounded-md border border-[#cbd5e1] bg-white text-[#0d253d] font-mono text-sm tabular-nums flex items-center justify-between gap-2.5 hover:border-[#94a3b8] focus:outline-none focus:border-[#533afd] focus:ring-1 focus:ring-[#533afd] transition-all disabled:opacity-50 cursor-pointer select-none"
                aria-label="Observation Date"
              >
                <span>{date}</span>
                <CalendarIcon className="size-4 text-[#64748d]" aria-hidden="true" />
              </button>
            </PopoverTrigger>
            <PopoverContent
              className="w-auto p-1 bg-white border border-[#e3e8ee] shadow-lg rounded-xl z-50"
              align="start"
            >
              <Calendar
                mode="single"
                selected={selectedDate}
                onSelect={handleSelectDate}
                defaultMonth={selectedDate}
                startMonth={new Date(OBSERVATION_DATE_RANGE.min)}
                endMonth={new Date(OBSERVATION_DATE_RANGE.max)}
                captionLayout="dropdown"
                className="rounded-lg"
              />
            </PopoverContent>
          </Popover>
        </div>

        {/* Latitude Input */}
        <div>
          <label
            htmlFor="target-lat-input"
            className="block text-[13px] font-medium text-[#64748d] mb-1 uppercase tracking-wider"
          >
            Latitude
          </label>
          <div className="relative flex items-center">
            <input
              id="target-lat-input"
              type="number"
              value={latitude}
              min={5.0}
              max={30.0}
              step={0.25}
              onChange={(e) => onLatitudeChange(parseFloat(e.target.value) || 0)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              className="h-9 w-24 pl-3 pr-8 rounded-md border border-[#cbd5e1] bg-white text-[#0d253d] font-mono text-sm tabular-nums focus:outline-none focus:border-[#533afd] focus:ring-1 focus:ring-[#533afd] transition-all disabled:opacity-50"
              aria-label="Target Latitude (5°–30°N)"
            />
            <span className="absolute right-2.5 text-[13px] font-mono text-[#64748d] pointer-events-none">
              °N
            </span>
          </div>
        </div>

        {/* Longitude Input */}
        <div>
          <label
            htmlFor="target-lon-input"
            className="block text-[13px] font-medium text-[#64748d] mb-1 uppercase tracking-wider"
          >
            Longitude
          </label>
          <div className="relative flex items-center">
            <input
              id="target-lon-input"
              type="number"
              value={longitude}
              min={45.0}
              max={105.0}
              step={0.25}
              onChange={(e) => onLongitudeChange(parseFloat(e.target.value) || 0)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              className="h-9 w-24 pl-3 pr-8 rounded-md border border-[#cbd5e1] bg-white text-[#0d253d] font-mono text-sm tabular-nums focus:outline-none focus:border-[#533afd] focus:ring-1 focus:ring-[#533afd] transition-all disabled:opacity-50"
              aria-label="Target Longitude (45°–105°E)"
            />
            <span className="absolute right-2.5 text-[13px] font-mono text-[#64748d] pointer-events-none">
              °E
            </span>
          </div>
        </div>

        {/* Station Presets Popover */}
        <div>
          <label
            className="block text-[13px] font-medium text-[#64748d] mb-1 uppercase tracking-wider"
          >
            Oceanographic Regime
          </label>
          <Popover open={presetsOpen} onOpenChange={setPresetsOpen}>
            <PopoverTrigger asChild>
              <button
                type="button"
                className="h-9 px-3.5 rounded-md border border-[#cbd5e1] bg-white hover:bg-[#f8fafc] text-sm text-[#273951] flex items-center justify-between gap-2 transition-colors cursor-pointer min-w-[200px]"
                title="Select a pre-calibrated oceanographic test station"
              >
                <span className="truncate">
                  {activePreset ? activePreset.name : 'Select Station Preset...'}
                </span>
                <ChevronDown className="size-3 text-[#64748d] shrink-0" />
              </button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-80 p-2 bg-white border-[#e3e8ee] text-sm shadow-xl rounded-xl">
              <div className="px-2.5 py-1 text-[13px] font-semibold uppercase tracking-wider text-[#64748d] border-b border-[#e2e8f0] mb-1.5">
                North Indian Ocean Regimes
              </div>
              <div className="space-y-1">
                {OCEAN_PRESETS.map((p) => {
                  const isSelected =
                    Math.abs(p.latitude - latitude) < 0.05 &&
                    Math.abs(p.longitude - longitude) < 0.05;
                  return (
                    <button
                      type="button"
                      key={p.name}
                      onClick={() => {
                        onSelectPreset(p);
                        setPresetsOpen(false);
                      }}
                      className={`w-full text-left p-2 rounded-lg flex items-start justify-between gap-2 transition-colors cursor-pointer ${
                        isSelected
                          ? 'bg-[#f1f5f9] text-[#0d253d] font-medium'
                          : 'hover:bg-[#f8fafc] text-[#273951]'
                      }`}
                    >
                      <div>
                        <div className="text-sm flex items-center gap-1.5">
                          <span>{p.name}</span>
                          {isSelected && <Check className="size-3 text-[#533afd]" />}
                        </div>
                        <div className="text-[13px] font-mono text-[#64748d] mt-0.5">
                          {p.latitude}°N, {p.longitude}°E · {p.region}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </PopoverContent>
          </Popover>
        </div>

        {/* Primary Reconstruction Action Button */}
        <div>
          <Button
            onClick={onReconstruct}
            disabled={loading || !isValid}
            className="h-9 px-5 text-sm rounded-full font-medium"
            aria-label="Reconstruct Subsurface Profile"
          >
            {loading ? (
              <>
                <Spinner data-icon="inline-start" />
                <span>Reconstructing…</span>
              </>
            ) : (
              <>
                <span>Reconstruct Profile</span>
                <span className="text-[13px] font-mono text-white/70 ml-1.5 hidden sm:inline">
                  ⌘↵
                </span>
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Domain Out-of-Bounds Warning */}
      {!isValid && (
        <div className="text-sm text-rose-700 font-mono py-1">
          Coordinates outside NIO domain (5.00°–30.00°N, 45.00°–105.00°E)
        </div>
      )}
    </div>
  );
};

