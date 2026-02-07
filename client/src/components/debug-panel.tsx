import { useMemo } from "react";
import type { VisualizationData, VisualizationPoint } from "@shared/schema";

interface DebugPanelProps {
  data: VisualizationData;
  currentTime: number;
}

export function DebugPanel({ data, currentTime }: DebugPanelProps) {
  const allPoints = useMemo(() => {
    return data.verses.flatMap(v => v.points);
  }, [data.verses]);

  const currentPoint = useMemo(() => {
    if (!allPoints.length) return null;
    
    let closest = allPoints[0];
    let minDiff = Math.abs(closest.time - currentTime);
    
    for (const point of allPoints) {
      const diff = Math.abs(point.time - currentTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = point;
      }
    }
    
    return closest;
  }, [allPoints, currentTime]);

  const currentVerse = useMemo(() => {
    return data.verses.find(v => currentTime >= v.start && currentTime <= v.end);
  }, [data.verses, currentTime]);

  const visiblePointCount = useMemo(() => {
    return allPoints.filter((p: VisualizationPoint) => p.time <= currentTime).length;
  }, [allPoints, currentTime]);

  const stats = useMemo(() => {
    const avgEnergy = allPoints.reduce((sum: number, p: VisualizationPoint) => sum + p.size, 0) / allPoints.length;
    return { avgEnergy };
  }, [allPoints]);

  return (
    <div 
      className="fixed top-20 right-4 z-10 w-72 p-4 backdrop-blur-xl bg-black/70 rounded-xl border border-white/10"
      data-testid="panel-debug"
    >
      <h3 className="text-xs uppercase tracking-wide text-muted-foreground mb-4 font-medium border-b border-white/10 pb-2">
        Debug Overlay
      </h3>

      <div className="space-y-4">
        <div className="space-y-2">
          <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
            Playback
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Time</span>
              <span className="font-mono text-sm text-primary">{currentTime.toFixed(2)}s</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Duration</span>
              <span className="font-mono text-sm text-foreground">{data.duration.toFixed(2)}s</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Progress</span>
              <span className="font-mono text-sm text-foreground">{((currentTime / data.duration) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Verse</span>
              <span className="font-mono text-sm text-foreground">{currentVerse ? `#${data.verses.indexOf(currentVerse) + 1}` : "-"}</span>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
            Visualization
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Total Points</span>
              <span className="font-mono text-sm text-foreground">{allPoints.length}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Visible</span>
              <span className="font-mono text-sm text-foreground">{visiblePointCount}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Verses</span>
              <span className="font-mono text-sm text-foreground">{data.verses.length}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Avg Energy</span>
              <span className="font-mono text-sm text-foreground">{stats.avgEnergy.toFixed(3)}</span>
            </div>
          </div>
        </div>

        {currentPoint && (
          <div className="space-y-2">
            <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
              Current Point (t={currentPoint.time.toFixed(3)}s)
            </h4>
            <div className="grid grid-cols-2 gap-2">
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">X (Time)</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.x.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Y (Pitch)</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.y.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Z (Centroid)</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.z.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Size (Energy)</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.size.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Beat Strength</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.beatStrength?.toFixed(3) ?? "0.000"}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Complexity</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.complexity?.toFixed(3) ?? "0.000"}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Band</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.band ?? "-"}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Onset (L/M/H)</span>
                <span className="font-mono text-sm text-foreground">
                  {currentPoint.onsetLow?.toFixed(2) ?? "0"}/{currentPoint.onsetMid?.toFixed(2) ?? "0"}/{currentPoint.onsetHigh?.toFixed(2) ?? "0"}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Band Offset Z</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.bandOffsetZ?.toFixed(3) ?? "0.000"}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
