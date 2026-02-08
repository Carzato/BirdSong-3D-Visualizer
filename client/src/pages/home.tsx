import { useState, useCallback } from "react";
import { UploadInterface } from "@/components/upload-interface";
import { VisualizationCanvas } from "@/components/visualization-canvas";
import { ControlPanel } from "@/components/control-panel";
import { FloatingSettings } from "@/components/floating-settings";
import { DebugPanel } from "@/components/debug-panel";
import { LoadingOverlay } from "@/components/loading-overlay";
import { MFCCPanel } from "@/components/mfcc-panel";
import { FrequencyScale } from "@/components/frequency-scale";
import type { VisualizationData, VisualizationSettings, VisualizationPoint } from "@shared/schema";

export default function Home() {
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("Analyzing audio...");
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState<VisualizationPoint | null>(null);
  const [settings, setSettings] = useState<VisualizationSettings>({
    visualStyle: "network",
    autoRotate: true,
    loopPlayback: true,
    showDebug: false,
    isFullscreen: false,
    progressiveReveal: true,
  });

  const handleUploadStart = useCallback(() => {
    setIsLoading(true);
    setLoadingMessage("Uploading audio file...");
  }, []);

  const handleAnalyzing = useCallback(() => {
    setLoadingMessage("Analyzing audio features...");
  }, []);

  const handleUploadComplete = useCallback((data: VisualizationData) => {
    setVisualizationData(data);
    setIsLoading(false);
    setCurrentTime(0);
    setIsPlaying(false);
  }, []);

  const handleUploadError = useCallback(() => {
    setIsLoading(false);
  }, []);

  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleSeek = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handlePlayPause = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  const handleSettingsChange = useCallback((newSettings: Partial<VisualizationSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const handleReset = useCallback(() => {
    setVisualizationData(null);
    setCurrentTime(0);
    setIsPlaying(false);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      handleSettingsChange({ isFullscreen: true });
    } else {
      document.exitFullscreen();
      handleSettingsChange({ isFullscreen: false });
    }
  }, [handleSettingsChange]);

  const handlePointHover = useCallback((point: VisualizationPoint | null) => {
    setSelectedPoint(point);
  }, []);

  if (!visualizationData) {
    return (
      <div className="relative min-h-screen bg-gradient-radial from-background via-background to-black flex items-center justify-center">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-gray-900/20 via-background to-black" />
        <UploadInterface
          onUploadStart={handleUploadStart}
          onAnalyzing={handleAnalyzing}
          onUploadComplete={handleUploadComplete}
          onUploadError={handleUploadError}
        />
        {isLoading && <LoadingOverlay message={loadingMessage} />}
      </div>
    );
  }

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black">
      <VisualizationCanvas
        data={visualizationData}
        currentTime={currentTime}
        isPlaying={isPlaying}
        settings={settings}
        onPointHover={handlePointHover}
      />

      <FrequencyScale />
      <MFCCPanel selectedPoint={selectedPoint} />

      <FloatingSettings
        settings={settings}
        onSettingsChange={handleSettingsChange}
        onToggleFullscreen={toggleFullscreen}
        onReset={handleReset}
      />

      {settings.showDebug && (
        <DebugPanel data={visualizationData} currentTime={currentTime} />
      )}

      <ControlPanel
        data={visualizationData}
        currentTime={currentTime}
        isPlaying={isPlaying}
        settings={settings}
        onTimeUpdate={handleTimeUpdate}
        onSeek={handleSeek}
        onPlayPause={handlePlayPause}
        onSettingsChange={handleSettingsChange}
      />

      {isLoading && <LoadingOverlay message={loadingMessage} />}
    </div>
  );
}
