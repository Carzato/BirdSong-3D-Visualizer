# BirdSong 3D Visualizer - Design Guidelines

## Design Approach

**Reference-Based: Premium Audio/Creative Tool Aesthetic**
Drawing inspiration from Spotify's audio UI patterns, Adobe Creative Cloud's professional interfaces, and WebGL showcase platforms like Shadertoy. The design prioritizes an immersive, dark canvas-first experience that puts the 3D visualization center stage.

**Core Principles:**
1. **Canvas Dominance** - 3D visualization occupies 90%+ of viewport, UI chrome is minimal and unobtrusive
2. **Dark Immersion** - Deep dark backgrounds (near-black) to maximize contrast for glowing visualization points
3. **Precision Controls** - Professional-grade controls that feel responsive and exact
4. **Adaptive Transparency** - UI elements use frosted glass/blur effects to float above visualization without blocking it

## Typography

**Font Stack:**
- Primary: Inter (via Google Fonts CDN) - clean, technical, modern
- Monospace: JetBrains Mono (for debug overlay, time codes)

**Hierarchy:**
- Page title/brand: 24px, weight 600 (only shown in corner or pre-upload state)
- Section labels: 14px, weight 500, tracking wide (0.05em)
- Control labels: 13px, weight 400
- Time codes/data: 12px, monospace
- All text in light gray (#E5E5E5) on dark backgrounds, never pure white

## Layout System

**Spacing Primitives:** Use Tailwind units of 2, 3, 4, 6, 8, 12
- Consistent 4-unit (1rem) base rhythm for most spacing
- 2-unit for tight groupings (button clusters)
- 8-12 units for major section separation

**Grid Structure:**
- Full-viewport canvas with absolute-positioned UI overlays
- Upload state: centered modal max-w-xl with 8-unit padding
- Main interface: Fixed control panel at bottom (h-20 to h-24), floating settings top-right

## Component Library

### A. Upload Interface (Initial State)
**Layout:** Centered card on gradient dark background
- Card: max-w-xl, rounded-2xl, backdrop-blur-xl, border border-white/10
- Drag-drop zone: min-h-64, dashed border, transition on hover
- File requirements: small text below, color-coded (green for supported formats)
- Upload button: Large, rounded-full, gradient from purple to blue
- Sample audio links: Grid of 2-3 preset bird songs below upload area

### B. Main Canvas
**3D Visualization Area:**
- position: fixed, inset-0, z-0
- Background: Radial gradient from very dark gray (#0A0A0A) center to pure black edges
- No borders or containment - edge-to-edge immersion

### C. Control Panel (Bottom Bar)
**Structure:** Fixed bottom, w-full, h-20, backdrop-blur-2xl, bg-black/40, border-t border-white/5

**Left Section (Play Controls):**
- Play/Pause button: 48px circle, gradient border, centered icon
- Time display: "00:23 / 01:45" in monospace, muted text

**Center Section (Timeline Scrubber):**
- Full-width progress bar with verse markers
- Track: h-1, rounded-full, bg-white/10
- Progress fill: gradient from cyan to purple, rounded-full
- Scrubber handle: 12px circle, white, drop-shadow-lg, appears on hover
- Verse dividers: vertical lines (h-3) with labels above on hover

**Right Section (Settings):**
- Style dropdown: rounded-lg button with chevron
- Fullscreen toggle: icon button
- All buttons: hover:bg-white/10, transition duration-200

### D. Floating Top Bar
**Position:** top-4, right-4, absolute, z-10

**Elements (horizontal flex, gap-3):**
- Color palette selector: 4 circular swatches in a row, 32px each, border-2 on active
- Dark/Light background toggle: icon button with moon/sun
- Loop playback toggle: icon button with repeat icon
- Auto-rotate toggle: icon button with rotation arrows
- All buttons: backdrop-blur-xl, bg-white/5, rounded-lg, p-2

### E. Verse Labels (Canvas Overlay)
**Rendering:** Positioned in 3D space above verse clusters
- Text: 16px, weight 600, white with text-shadow for legibility
- Background: None or subtle dark pill on hover
- Connected with thin line (1px, white/30%) to verse start point

### F. Debug Overlay (Optional Toggle)
**Position:** top-4, left-4, w-64
**Style:** backdrop-blur-xl, bg-black/60, rounded-xl, p-4, border border-white/10

**Content:**
- Feature values in 2-column grid
- Labels: 11px, uppercase, tracking-wide, text-gray-400
- Values: 14px, monospace, text-cyan-400
- Updates in real-time with playback

### G. Loading State
**Overlay:** Full-screen, backdrop-blur-xl, bg-black/80
**Spinner:** Custom visualization loading animation (rotating points forming a circle)
**Text:** "Analyzing audio..." below spinner, 16px

### H. Error States
**Toast notification:** top-4, centered, slide down animation
- Container: rounded-xl, bg-red-500/90, backdrop-blur-sm, p-4, max-w-md
- Icon: X or alert circle, 20px
- Message: 14px, white
- Auto-dismiss after 5s with progress bar

## Color Palette System

**Preset Schemes (4 options):**
1. **Aurora** - Gradient from cyan (#00D9FF) through purple (#A000FF) to magenta (#FF00E5)
2. **Ocean Depths** - Deep blue (#0047AB) to teal (#00CED1) to seafoam (#3FE0D0)
3. **Solar Flare** - Orange (#FF6B35) to yellow (#F7B32B) to white (#FFE66D)
4. **Monochrome** - White to gray gradients with subtle blue tint

**Application:**
- Point colors derived from spectral centroid mapped to chosen palette
- Edge colors: Same as points but at 30% opacity
- Active/highlighted points: +50% brightness with glow effect

## Visual Effects

**Post-Processing:**
- Subtle bloom on points (radius 2-3px) for glow effect
- Anti-aliasing enabled for smooth edges on hi-res displays
- Ambient camera motion: Slow orbital drift at 0.1 rotation/minute when paused

**Transitions:**
- UI elements: duration-200 for hovers, duration-300 for state changes
- 3D scene: Smooth camera movements with easing
- Point highlighting: Fast fade-in (100ms) when approaching playback time

## Accessibility
- Keyboard shortcuts: Space for play/pause, F for fullscreen, Arrow keys for scrubbing
- Focus states: 2px white outline with offset
- Contrast: Maintain 4.5:1 minimum on all text
- Screen reader labels on all icon-only buttons

## Images
**No hero images required.** This application is canvas-first. The only imagery is:
- Optional: Small thumbnail icons for preset audio samples in upload state
- Brand/logo mark in corner (if applicable) - simple, minimal, monochrome