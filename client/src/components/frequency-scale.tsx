export function FrequencyScale() {
  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-black/70 px-6 py-2 rounded-lg border border-gray-700">
      <div className="text-xs font-medium text-gray-300 uppercase tracking-wide">Frequency:</div>
      <div className="flex gap-3 items-center">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(0, 80%, 60%)' }}></div>
          <span className="text-xs text-red-400">0-2.5 kHz</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(30, 80%, 60%)' }}></div>
          <span className="text-xs text-orange-400">2.5-5 kHz</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(45, 80%, 60%)' }}></div>
          <span className="text-xs text-yellow-400">5-7.5 kHz</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(60, 80%, 60%)' }}></div>
          <span className="text-xs text-yellow-200">7.5-10 kHz</span>
        </div>
      </div>
    </div>
  );
}
