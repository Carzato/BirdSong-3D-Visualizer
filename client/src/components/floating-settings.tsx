import { Repeat, Move3D, Bug, Maximize, Upload, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { VisualizationSettings } from "@shared/schema";

interface FloatingSettingsProps {
  settings: VisualizationSettings;
  onSettingsChange: (settings: Partial<VisualizationSettings>) => void;
  onToggleFullscreen: () => void;
  onReset: () => void;
}

export function FloatingSettings({
  settings,
  onSettingsChange,
  onToggleFullscreen,
  onReset,
}: FloatingSettingsProps) {
  return (
    <div 
      className="fixed top-4 right-4 z-10 flex items-center gap-3"
      data-testid="panel-floating-settings"
    >
      <div className="flex items-center gap-1 p-1 backdrop-blur-xl bg-white/5 rounded-lg border border-white/10">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              className={`w-8 h-8 ${settings.progressiveReveal ? "text-primary" : "text-muted-foreground"}`}
              onClick={() => onSettingsChange({ progressiveReveal: !settings.progressiveReveal })}
              aria-label="Toggle progressive reveal"
              data-testid="button-progressive"
            >
              <Wand2 className="w-4 h-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Live reveal {settings.progressiveReveal ? "(on)" : "(off)"}</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              className={`w-8 h-8 ${settings.loopPlayback ? "text-primary" : "text-muted-foreground"}`}
              onClick={() => onSettingsChange({ loopPlayback: !settings.loopPlayback })}
              aria-label="Toggle loop playback"
              data-testid="button-loop"
            >
              <Repeat className="w-4 h-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Loop playback {settings.loopPlayback ? "(on)" : "(off)"}</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              className={`w-8 h-8 ${settings.autoRotate ? "text-primary" : "text-muted-foreground"}`}
              onClick={() => onSettingsChange({ autoRotate: !settings.autoRotate })}
              aria-label="Toggle auto-rotate"
              data-testid="button-auto-rotate"
            >
              <Move3D className="w-4 h-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Auto-rotate {settings.autoRotate ? "(on)" : "(off)"}</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              className={`w-8 h-8 ${settings.showDebug ? "text-primary" : "text-muted-foreground"}`}
              onClick={() => onSettingsChange({ showDebug: !settings.showDebug })}
              aria-label="Toggle debug panel"
              data-testid="button-debug"
            >
              <Bug className="w-4 h-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Debug overlay {settings.showDebug ? "(on)" : "(off)"}</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              className="w-8 h-8"
              onClick={onToggleFullscreen}
              aria-label="Toggle fullscreen"
              data-testid="button-fullscreen"
            >
              <Maximize className="w-4 h-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Fullscreen (F)</p>
          </TooltipContent>
        </Tooltip>
      </div>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon"
            variant="ghost"
            className="w-8 h-8 backdrop-blur-xl bg-white/5 border border-white/10"
            onClick={onReset}
            aria-label="Upload new audio"
            data-testid="button-new-upload"
          >
            <Upload className="w-4 h-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <p>Upload new audio</p>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
