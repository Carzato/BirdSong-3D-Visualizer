# BirdSong 3D Visualizer

## Overview
A web-based audio visualization application that transforms uploaded audio files (WAV, MP3, M4A) into interactive 3D point network visualizations using spectral analysis and Three.js. The application provides an immersive, hi-res display-optimized experience with real-time synchronized playback, frequency-based coloring, and professional-grade visual quality.

## Current State
MVP complete with full audio analysis pipeline and 3D visualization.

## Recent Changes
- 2026-01-12: Per-band onset detection with spatial separation
  - Added computeBandEnergies() for low (20-250Hz), mid (250-2kHz), high (2-8kHz) bands
  - Added computeBandOnsets() using spectral flux with per-band normalization
  - pickDominantBand() selects dominant band and corresponding onset per frame
  - Points now have band, bandOffsetZ, onset fields for per-band activation
  - bandOffsetZ creates spatial separation: low=-0.2, mid=0, high=+0.2
  - Tighter 0.06s activation window with onset-driven pulsing
  - Debug panel shows Band, Onset, and Band Offset Z values
- 2026-01-12: Debug overlay and frequency mapping verification
  - Fixed debug overlay to show real-time playback and point data on right side
  - Clear axis labels: X (Time), Y (Pitch), Z (Centroid), Size (Energy)
  - Shows beat strength, complexity, verse info, and visible point count
  - Increased EMA smoothing alpha to 0.5 for better responsiveness
- 2026-01-12: Enhanced visual smoothness for complex music
  - Temporal EMA smoothing (alpha=0.5) on audio features in analyzer
  - Target/current interpolation for size and brightness with clamped max delta
  - Immutable basePositions array prevents drift accumulation
  - Subtle simplex noise drift so points feel alive during quiet sections
  - Eased auto-orbit camera with smooth damping
  - Widened highlight window to 0.15s for smoother activation transitions
- 2026-01-12: Added beat detection and audio complexity analysis
  - Spectral flux analysis for beat/onset detection
  - Adaptive peak-picking algorithm with dynamic thresholding
  - Per-point beatStrength drives visual pulsing synchronized to music beats
  - Spectral flatness measures instrument complexity
  - Points pulse dramatically on detected beats with size and brightness boost
- 2026-01-12: Replaced color palettes with frequency-based coloring
  - Point colors now determined by spectral centroid (audio frequency content)
  - Blue/purple (2-4kHz low freq) → Green/cyan (4-6kHz mid) → Yellow/red (6-8kHz high)
  - Removed color palette selector from UI
  - Network lines use neutral blue color for clarity
- 2026-01-12: Added progressive "live reveal" visualization mode
  - Points and network lines now appear in sync with audio playback
  - Toggle button (wand icon) in settings bar to enable/disable
  - Enabled by default for immersive experience
  - Edge buffer built in time-sorted order for correct reveal with drawRange
- 2026-01-12: Added MP3/M4A support with ffmpeg transcoding
  - Increased max file size from 20MB to 300MB
  - Added fluent-ffmpeg for server-side audio transcoding
  - MP3 and M4A files are automatically converted to WAV for analysis
  - Temporary transcoded files are cleaned up after processing
  - Duration detection via ffprobe for compressed formats
- 2026-01-12: Initial implementation complete
  - WAV-only file upload with validation
  - Real PCM-based audio analysis (RMS, ZCR pitch, spectral centroid/bandwidth)
  - Silence-based verse segmentation
  - 3D point network visualization with k-nearest neighbor edges
  - Synchronized playback with real-time highlighting
  - Dark immersive theme following design guidelines

## Project Architecture

### Frontend (client/src/)
- **pages/home.tsx**: Main page with upload interface and visualization canvas
- **components/upload-interface.tsx**: Drag-drop file upload with validation
- **components/visualization-canvas.tsx**: Three.js/React Three Fiber 3D visualization
- **components/control-panel.tsx**: Playback controls (play/pause, scrubber, time display)
- **components/floating-settings.tsx**: Floating settings bar for visualization controls
- **components/debug-overlay.tsx**: Performance metrics and verse info display
- **lib/visualizationTypes.ts**: Shared TypeScript types

### Backend (server/)
- **routes.ts**: Express routes for file upload, analysis, and static serving
- **audio-analyzer.ts**: WAV parsing, PCM extraction, spectral feature analysis

### Shared (shared/)
- **schema.ts**: Zod schemas and TypeScript types for visualization data

## Key Technical Decisions
- Server-side ffmpeg transcoding for MP3/M4A to ensure authentic PCM-based analysis for all formats
- React Three Fiber for declarative Three.js integration
- In-memory storage for uploaded files (no database required for this MVP)
- Spectral features: RMS amplitude, zero-crossing rate for pitch estimation, spectral centroid and bandwidth via DFT
- Temporary file cleanup after transcoding to prevent disk usage accumulation

## File Upload Constraints
- WAV, MP3, M4A formats supported
- Max 10 minutes duration (600 seconds)
- Max 300MB file size

## Running the Project
The project runs via `npm run dev` which starts both Express backend and Vite frontend on port 5000.
