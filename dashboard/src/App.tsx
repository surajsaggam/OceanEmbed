import { useEffect } from 'react';
import { AlertCircle, X } from 'lucide-react';
import { AppHeader } from './components/shell/AppHeader';
import { TargetingStationBar } from './components/targeting/TargetingStationBar';
import { AnalysisWorkbench } from './components/workbench/AnalysisWorkbench';
import { useOceanEmbed } from './hooks/useOceanEmbed';
import type { OceanPreset } from './types/api';

export function App() {
  const {
    date,
    setDate,
    latitude,
    setLatitude,
    longitude,
    setLongitude,
    loading,
    error,
    clearError,
    health,
    reconstruction,
    scatterData,
    executeReconstruction,
    selectPreset,
  } = useOceanEmbed({
    initialDate: '2023-06-15',
    initialLatitude: 18.5,
    initialLongitude: 88.25,
    autoReconstructOnMount: true,
  });

  // Global accelerator shortcut: Cmd+Enter or Ctrl+Enter to trigger reconstruction
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        executeReconstruction();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [executeReconstruction]);

  const handleSelectCoordinates = (lat: number, lon: number) => {
    setLatitude(lat);
    setLongitude(lon);
    executeReconstruction(lat, lon, date);
  };

  const handleSelectPreset = (preset: OceanPreset) => {
    selectPreset(preset);
  };

  return (
    <div className="h-screen bg-[#f6f9fc] text-[#0d253d] flex flex-col font-sans selection:bg-[#533afd]/15 selection:text-[#533afd] overflow-hidden">
      {/* Global Application Header */}
      <AppHeader health={health} reconstruction={reconstruction} />

      {/* Targeting Station & Date Specification Bar */}
      <TargetingStationBar
        date={date}
        latitude={latitude}
        longitude={longitude}
        loading={loading}
        onDateChange={setDate}
        onLatitudeChange={setLatitude}
        onLongitudeChange={setLongitude}
        onReconstruct={() => executeReconstruction()}
        onSelectPreset={handleSelectPreset}
      />

      {/* Runtime / API Error Banner */}
      {error && (
        <div
          role="alert"
          className="mx-6 my-2 px-4 py-2 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm font-mono flex items-center justify-between gap-3 shadow-xs animate-in fade-in-0 duration-150"
        >
          <div className="flex items-center gap-2.5">
            <AlertCircle className="size-4 shrink-0 text-rose-600" />
            <span>
              <b>API Diagnostic:</b> {error}
            </span>
          </div>
          <button
            type="button"
            onClick={clearError}
            className="text-rose-600 hover:text-rose-900 transition-colors cursor-pointer"
            aria-label="Dismiss error"
          >
            <X className="size-4" />
          </button>
        </div>
      )}

      {/* Main Analysis Workbench — fills remaining viewport height, scrolls internally */}
      <div className="flex-1 overflow-y-auto">
        <AnalysisWorkbench
          latitude={latitude}
          longitude={longitude}
          loading={loading}
          reconstruction={reconstruction}
          scatterData={scatterData}
          onSelectCoordinates={handleSelectCoordinates}
        />
      </div>
    </div>
  );
}

export default App;
