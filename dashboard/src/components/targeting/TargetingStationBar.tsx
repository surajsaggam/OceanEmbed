import React, { useState, useMemo } from 'react';
import { CalendarIcon, FileDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import { Calendar } from '@/components/ui/calendar';
import { Popover, PopoverTrigger, PopoverContent } from '@/components/ui/popover';
import { ComboBox, Input, Label, ListBox, Header } from '@heroui/react';
import { OCEAN_PRESETS } from '@/data/presets';
import { isWithinNioDomain, OBSERVATION_DATE_RANGE } from '@/lib/ocean';
import type { OceanPreset, ReconstructionHistoryItem } from '@/types/api';

interface TargetingStationBarProps {
  date: string;
  latitude: number;
  longitude: number;
  loading: boolean;
  history?: ReconstructionHistoryItem[];
  hasReconstruction?: boolean;
  isGeneratingReport?: boolean;
  onGenerateReport?: () => void;
  onDateChange: (d: string) => void;
  onLatitudeChange: (lat: number) => void;
  onLongitudeChange: (lon: number) => void;
  onReconstruct: () => void;
  onSelectPreset: (preset: OceanPreset) => void;
  onSelectHistoryItem?: (item: ReconstructionHistoryItem) => void;
}

export const TargetingStationBar: React.FC<TargetingStationBarProps> = ({
  date,
  latitude,
  longitude,
  loading,
  history = [],
  hasReconstruction = false,
  isGeneratingReport = false,
  onGenerateReport,
  onDateChange,
  onLatitudeChange,
  onLongitudeChange,
  onReconstruct,
  onSelectPreset,
  onSelectHistoryItem,
}) => {

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

  const activeHistoryItem = history.find(
    (h) =>
      Math.abs(h.latitude - latitude) < 0.05 &&
      Math.abs(h.longitude - longitude) < 0.05 &&
      h.date === date
  );

  const selectedKey = activePreset
    ? activePreset.name
    : activeHistoryItem
    ? `history_${activeHistoryItem.id}`
    : null;

  const handleRegimeChange = (key: React.Key | null) => {
    if (!key) return;
    const keyStr = String(key);
    if (keyStr.startsWith('history_')) {
      const histId = parseInt(keyStr.replace('history_', ''), 10);
      const histItem = history.find((h) => h.id === histId);
      if (histItem && onSelectHistoryItem) {
        onSelectHistoryItem(histItem);
      }
      return;
    }
    const preset = OCEAN_PRESETS.find((p) => p.name === keyStr);
    if (preset) {
      onSelectPreset(preset);
    }
  };


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
            className="block text-[11px] font-semibold text-[#64748d] mb-1.5 uppercase tracking-wider"
          >
            Observation Date
          </label>
          <Popover open={datePickerOpen} onOpenChange={setDatePickerOpen}>
            <PopoverTrigger asChild>
              <button
                id="target-date-trigger"
                type="button"
                disabled={loading}
                className="h-9 px-3 rounded-lg border border-[#cbd5e1] bg-white text-[#0d253d] font-mono text-sm tabular-nums flex items-center justify-between gap-2.5 hover:border-[#94a3b8] focus:outline-none focus:border-[#533afd] focus:ring-1 focus:ring-[#533afd] transition-all disabled:opacity-50 cursor-pointer select-none"
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
            className="block text-[11px] font-semibold text-[#64748d] mb-1.5 uppercase tracking-wider"
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
              className="h-9 w-24 pl-3 pr-8 rounded-lg border border-[#cbd5e1] bg-white text-[#0d253d] font-mono text-sm tabular-nums focus:outline-none focus:border-[#533afd] focus:ring-1 focus:ring-[#533afd] transition-all disabled:opacity-50"
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
            className="block text-[11px] font-semibold text-[#64748d] mb-1.5 uppercase tracking-wider"
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
              className="h-9 w-24 pl-3 pr-8 rounded-lg border border-[#cbd5e1] bg-white text-[#0d253d] font-mono text-sm tabular-nums focus:outline-none focus:border-[#533afd] focus:ring-1 focus:ring-[#533afd] transition-all disabled:opacity-50"
              aria-label="Target Longitude (45°–105°E)"
            />
            <span className="absolute right-2.5 text-[13px] font-mono text-[#64748d] pointer-events-none">
              °E
            </span>
          </div>
        </div>

        {/* Oceanographic Regime HeroUI ComboBox */}
        <div>
          <ComboBox
            selectedKey={selectedKey}
            onSelectionChange={handleRegimeChange}
            className="w-60 sm:w-72"
          >
            <Label className="block text-[11px] font-semibold text-[#64748d] mb-1.5 uppercase tracking-wider">
              Oceanographic Regime
            </Label>
            <ComboBox.InputGroup className="h-9 px-3 rounded-lg border border-[#cbd5e1] bg-white flex items-center justify-between gap-2 text-sm text-[#273951] hover:border-[#94a3b8] focus-within:border-[#533afd] focus-within:ring-1 focus-within:ring-[#533afd] transition-all">
              <Input
                placeholder="Select Station Preset..."
                className="w-full bg-transparent text-sm text-[#0d253d] font-normal outline-none placeholder:text-[#94a3b8]"
              />
              <ComboBox.Trigger className="text-[#64748d] hover:text-[#0d253d] transition-colors cursor-pointer shrink-0 flex items-center justify-center p-0.5" />
            </ComboBox.InputGroup>
            <ComboBox.Popover className="min-w-[340px] w-96 bg-white border border-[#e3e8ee] shadow-xl rounded-xl p-1 z-50">
              <ListBox className="outline-none space-y-2 p-1 max-h-[380px] overflow-y-auto">
                {/* Station Presets Section */}
                <ListBox.Section key="presets-section">
                  <Header className="px-2.5 py-1 text-[10px] font-bold text-[#64748d] uppercase tracking-wider bg-slate-50/90 rounded border-b border-slate-100 mb-1 flex items-center justify-between">
                    <span>Station Presets</span>
                    <span className="text-[10px] font-mono text-[#94a3b8] font-normal">{OCEAN_PRESETS.length} stations</span>
                  </Header>
                  {OCEAN_PRESETS.map((p) => (
                    <ListBox.Item
                      key={p.name}
                      id={p.name}
                      textValue={p.name}
                      className="w-full text-left p-2.5 rounded-lg flex items-start justify-between gap-2 transition-colors cursor-pointer outline-none hover:bg-[#f8fafc] data-[selected=true]:bg-[#f1f5f9] data-[focused=true]:bg-[#f8fafc]"
                    >
                      <div className="flex flex-col flex-1 min-w-0">
                        <span className="text-sm font-medium text-[#0d253d]">
                          {p.name}
                        </span>
                        <span className="text-[13px] font-mono text-[#64748d] mt-0.5">
                          {p.latitude}°N, {p.longitude}°E · {p.region}
                        </span>
                      </div>
                      <ListBox.ItemIndicator className="text-[#533afd] size-4 shrink-0 mt-0.5" />
                    </ListBox.Item>
                  ))}
                </ListBox.Section>

                {/* Recent Reconstructions Section */}
                <ListBox.Section key="history-section" className="border-t border-[#e2e8f0] pt-2 mt-1">
                  <Header className="px-2.5 py-1 text-[10px] font-bold text-[#64748d] uppercase tracking-wider bg-slate-50/90 rounded border-b border-slate-100 mb-1 flex items-center justify-between">
                    <span>Recent Reconstructions</span>
                    {history && history.length > 0 && (
                      <span className="text-[10px] font-mono text-[#94a3b8] font-normal">
                        {history.length} {history.length === 1 ? 'entry' : 'entries'}
                      </span>
                    )}
                  </Header>
                  {history && history.length > 0 ? (
                    history.map((item) => (
                      <ListBox.Item
                        key={`history_${item.id}`}
                        id={`history_${item.id}`}
                        textValue={`${item.regime} ${item.date}`}
                        className="w-full text-left p-2.5 rounded-lg flex items-start justify-between gap-2 transition-colors cursor-pointer outline-none hover:bg-[#f8fafc] data-[selected=true]:bg-[#f1f5f9] data-[focused=true]:bg-[#f8fafc]"
                      >
                        <div className="flex flex-col flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-sm font-medium text-[#0d253d] truncate">
                              {item.regime}
                            </span>
                            <span className="text-[11px] font-mono text-[#64748d] shrink-0 font-normal">
                              {item.date}
                            </span>
                          </div>
                          <span className="text-[12px] font-mono text-[#64748d] mt-0.5">
                            {item.latitude.toFixed(2)}°N, {item.longitude.toFixed(2)}°E
                          </span>
                        </div>
                        <ListBox.ItemIndicator className="text-[#533afd] size-4 shrink-0 mt-0.5" />
                      </ListBox.Item>
                    ))
                  ) : (
                    <ListBox.Item
                      id="empty-history"
                      isDisabled
                      textValue="No recent reconstructions"
                      className="w-full p-2.5 text-xs text-[#94a3b8] italic pointer-events-none"
                    >
                      No recent reconstructions yet
                    </ListBox.Item>
                  )}
                </ListBox.Section>
              </ListBox>
            </ComboBox.Popover>
          </ComboBox>
        </div>


        {/* Primary Reconstruction Action Button */}
        <div>
          <Button
            onClick={onReconstruct}
            disabled={loading || !isValid}
            className="h-9 px-5 text-sm rounded-full font-medium active:scale-[0.985] transition-all cursor-pointer"
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

        {/* Generate Report Action Button */}
        {hasReconstruction && (
          <div>
            <Button
              type="button"
              variant="outline"
              onClick={onGenerateReport}
              disabled={isGeneratingReport || loading}
              className="h-9 px-4 text-sm rounded-full font-medium border-[#cbd5e1] hover:border-[#533afd] hover:text-[#533afd] bg-white text-[#0d253d] active:scale-[0.985] transition-all cursor-pointer flex items-center gap-2 shadow-xs"
              aria-label="Generate Technical PDF Report"
            >
              {isGeneratingReport ? (
                <>
                  <Spinner data-icon="inline-start" />
                  <span>Generating PDF…</span>
                </>
              ) : (
                <>
                  <FileDown className="size-4 text-[#533afd]" aria-hidden="true" />
                  <span>Generate Report</span>
                </>
              )}
            </Button>
          </div>
        )}
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

