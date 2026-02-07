import { useState, useCallback, useRef } from "react";
import { Upload, Music, FileAudio, AlertCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import type { VisualizationData } from "@shared/schema";

interface UploadInterfaceProps {
  onUploadStart: () => void;
  onAnalyzing: () => void;
  onUploadComplete: (data: VisualizationData) => void;
  onUploadError: () => void;
}

const MAX_FILE_SIZE = 300 * 1024 * 1024;
const MAX_DURATION = 600;
const ACCEPTED_TYPES = ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a", "audio/m4a"];

export function UploadInterface({
  onUploadStart,
  onAnalyzing,
  onUploadComplete,
  onUploadError,
}: UploadInterfaceProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const validateFile = useCallback((file: File): Promise<boolean> => {
    return new Promise((resolve) => {
      if (file.size > MAX_FILE_SIZE) {
        setError(`File too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB`);
        resolve(false);
        return;
      }

      const isAccepted = ACCEPTED_TYPES.some(type => 
        file.type === type || 
        file.name.toLowerCase().endsWith('.wav') ||
        file.name.toLowerCase().endsWith('.mp3') ||
        file.name.toLowerCase().endsWith('.m4a')
      );

      if (!isAccepted) {
        setError("Unsupported file format. Please upload WAV, MP3, or M4A files.");
        resolve(false);
        return;
      }

      const audio = new Audio();
      audio.src = URL.createObjectURL(file);
      audio.onloadedmetadata = () => {
        URL.revokeObjectURL(audio.src);
        if (audio.duration > MAX_DURATION) {
          setError(`Audio too long. Maximum duration is ${MAX_DURATION} seconds.`);
          resolve(false);
        } else {
          setError(null);
          resolve(true);
        }
      };
      audio.onerror = () => {
        URL.revokeObjectURL(audio.src);
        setError("Could not read audio file. Please try a different file.");
        resolve(false);
      };
    });
  }, []);

  const uploadFile = useCallback(async (file: File) => {
    const isValid = await validateFile(file);
    if (!isValid) {
      onUploadError();
      return;
    }

    onUploadStart();

    const formData = new FormData();
    formData.append("audio", file);

    try {
      setTimeout(() => onAnalyzing(), 500);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || "Failed to analyze audio");
      }

      onUploadComplete(result.data);
      toast({
        title: "Analysis complete",
        description: "Your audio visualization is ready!",
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to upload file";
      setError(message);
      toast({
        title: "Upload failed",
        description: message,
        variant: "destructive",
      });
      onUploadError();
    }
  }, [validateFile, onUploadStart, onAnalyzing, onUploadComplete, onUploadError, toast]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  }, [uploadFile]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
  }, [uploadFile]);

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <Card
      className="relative z-10 max-w-xl w-full mx-4 p-8 backdrop-blur-xl bg-card/80 border-white/10 rounded-2xl"
      data-testid="card-upload"
    >
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-visualization-cyan via-visualization-purple to-visualization-magenta mb-4">
          <Music className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-2xl font-semibold text-foreground mb-2">
          BirdSong 3D Visualizer
        </h1>
        <p className="text-muted-foreground text-sm">
          Upload any audio file and watch it transform into living art
        </p>
      </div>

      <div
        className={`
          relative min-h-64 border-2 border-dashed rounded-xl
          flex flex-col items-center justify-center gap-4 p-8
          transition-all duration-200 cursor-pointer
          ${isDragging 
            ? "border-primary bg-primary/10 scale-[1.02]" 
            : "border-white/20 hover:border-white/40 hover:bg-white/5"
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        data-testid="dropzone-audio"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.m4a,audio/*"
          className="hidden"
          onChange={handleFileChange}
          data-testid="input-audio-file"
        />

        <div className={`
          p-4 rounded-full transition-all duration-200
          ${isDragging ? "bg-primary/20" : "bg-white/5"}
        `}>
          <Upload className={`w-8 h-8 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
        </div>

        <div className="text-center">
          <p className="text-foreground font-medium mb-1">
            {isDragging ? "Drop your audio file here" : "Drag & drop your audio file"}
          </p>
          <p className="text-muted-foreground text-sm">
            or click to browse
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <FileAudio className="w-3 h-3" />
            WAV, MP3, M4A
          </span>
          <span>Max 10 minutes</span>
          <span>Max 300MB</span>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-3 rounded-lg bg-destructive/10 border border-destructive/20 flex items-start gap-2" data-testid="alert-upload-error">
          <AlertCircle className="w-4 h-4 text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      <div className="mt-6 pt-6 border-t border-white/10">
        <p className="text-xs text-muted-foreground text-center mb-4 uppercase tracking-wide font-medium">
          Or try a sample
        </p>
        <div className="grid grid-cols-3 gap-3">
          {[
            { name: "Bird Song", icon: "🐦" },
            { name: "Forest", icon: "🌲" },
            { name: "Ocean", icon: "🌊" },
          ].map((sample) => (
            <Button
              key={sample.name}
              variant="outline"
              className="flex flex-col gap-1 h-auto py-3 border-white/10 hover:border-white/30"
              onClick={() => {
                toast({
                  title: "Sample audio",
                  description: "Sample audio files will be available in the next update.",
                });
              }}
              data-testid={`button-sample-${sample.name.toLowerCase().replace(" ", "-")}`}
            >
              <span className="text-lg">{sample.icon}</span>
              <span className="text-xs">{sample.name}</span>
            </Button>
          ))}
        </div>
      </div>
    </Card>
  );
}
