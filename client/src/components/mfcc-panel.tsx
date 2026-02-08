import type { VisualizationPoint } from "../../../shared/schema";

interface MFCCPanelProps {
  selectedPoint: VisualizationPoint | null;
}

export function MFCCPanel({ selectedPoint }: MFCCPanelProps) {
  if (!selectedPoint) {
    return (
      <div className="absolute right-4 top-20 bg-black/80 p-4 rounded-lg border border-gray-700">
        <h3 className="text-white mb-2 text-sm font-medium">MFCC Coefficients</h3>
        <p className="text-gray-400 text-xs">Select a point to view MFCC values</p>
      </div>
    );
  }

  return (
    <div className="absolute right-4 top-20 bg-black/90 p-4 rounded-lg border border-gray-700 max-h-[70vh] overflow-y-auto">
      <h3 className="text-white mb-3 text-sm font-medium">MFCC Coefficients</h3>
      <div className="text-xs text-gray-400 mb-3">
        Time: {selectedPoint.time.toFixed(2)}s
        {selectedPoint.frequency && (
          <span className="ml-3">Freq: {(selectedPoint.frequency / 1000).toFixed(2)} kHz</span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {selectedPoint.mfccs.map((value, i) => (
          <div key={i} className="flex justify-between items-center">
            <span className="text-gray-400">MFCC{String(i + 1).padStart(2, '0')}:</span>
            <span className="text-white font-mono">{value.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
