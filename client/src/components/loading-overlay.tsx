import { useEffect, useState } from "react";

interface LoadingOverlayProps {
  message?: string;
}

export function LoadingOverlay({ message = "Analyzing audio..." }: LoadingOverlayProps) {
  const [dots, setDots] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev + 1) % 4);
    }, 400);
    return () => clearInterval(interval);
  }, []);

  return (
    <div 
      className="fixed inset-0 z-50 flex flex-col items-center justify-center backdrop-blur-xl bg-black/80"
      data-testid="overlay-loading"
    >
      <div className="relative w-24 h-24 mb-6">
        <div className="absolute inset-0 flex items-center justify-center">
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              className="absolute w-3 h-3 rounded-full bg-gradient-to-r from-visualization-cyan to-visualization-purple"
              style={{
                transform: `rotate(${i * 45}deg) translateY(-32px)`,
                animation: `pulse 1.5s ease-in-out ${i * 0.1}s infinite`,
                opacity: 0.3 + (i * 0.08),
              }}
            />
          ))}
        </div>

        <div className="absolute inset-0 animate-spin" style={{ animationDuration: "3s" }}>
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-visualization-cyan shadow-lg shadow-visualization-cyan/50" />
        </div>
      </div>

      <p className="text-foreground text-base font-medium">
        {message}
        <span className="inline-block w-8 text-left">
          {".".repeat(dots)}
        </span>
      </p>

      <p className="text-muted-foreground text-sm mt-2">
        This may take a few seconds
      </p>

      <style>{`
        @keyframes pulse {
          0%, 100% {
            transform: rotate(var(--rotation)) translateY(-32px) scale(1);
            opacity: 0.3;
          }
          50% {
            transform: rotate(var(--rotation)) translateY(-32px) scale(1.3);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
