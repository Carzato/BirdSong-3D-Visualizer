import { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import type { VisualizationData, VisualizationSettings, VisualizationPoint } from "@shared/schema";

interface VisualizationCanvasProps {
  data: VisualizationData;
  currentTime: number;
  isPlaying: boolean;
  settings: VisualizationSettings;
  onPointHover?: (point: VisualizationPoint | null) => void;
}

interface SortedPoint {
  originalIndex: number;
  time: number;
}

function mapFrequencyToColor(normalizedFrequency: number): THREE.Color {
  const t = Math.max(0, Math.min(1, normalizedFrequency));
  
  if (t < 0.33) {
    const localT = t / 0.33;
    return new THREE.Color().setRGB(
      0.2 + localT * 0.1,
      0.1 + localT * 0.6,
      0.9 - localT * 0.3
    );
  } else if (t < 0.66) {
    const localT = (t - 0.33) / 0.33;
    return new THREE.Color().setRGB(
      0.3 + localT * 0.5,
      0.7 + localT * 0.2,
      0.6 - localT * 0.5
    );
  } else {
    const localT = (t - 0.66) / 0.34;
    return new THREE.Color().setRGB(
      0.8 + localT * 0.2,
      0.9 - localT * 0.5,
      0.1 + localT * 0.1
    );
  }
}

interface PointsVisualizationProps {
  data: VisualizationData;
  currentTime: number;
  settings: VisualizationSettings;
}

function simplex2D(x: number, y: number): number {
  const F2 = 0.5 * (Math.sqrt(3) - 1);
  const G2 = (3 - Math.sqrt(3)) / 6;
  
  const s = (x + y) * F2;
  const i = Math.floor(x + s);
  const j = Math.floor(y + s);
  const t = (i + j) * G2;
  const X0 = i - t;
  const Y0 = j - t;
  const x0 = x - X0;
  const y0 = y - Y0;
  
  const hash = (xi: number, yi: number) => {
    const n = xi + yi * 57;
    return Math.sin(n * 12.9898 + n * 78.233) * 43758.5453 % 1;
  };
  
  return (hash(i, j) * x0 + hash(i + 1, j) * y0) * 0.5;
}

function PointsVisualization({ data, currentTime, settings }: PointsVisualizationProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  const maxRevealedRef = useRef<number>(0);
  const currentSizesRef = useRef<Float32Array | null>(null);
  const currentBrightnessRef = useRef<Float32Array | null>(null);
  const timeRef = useRef<number>(0);

  const allPoints = useMemo(() => {
    return data.verses.flatMap(verse => verse.points);
  }, [data.verses]);

  const sortedPointIndices = useMemo(() => {
    const sorted: SortedPoint[] = allPoints.map((point, i) => ({
      originalIndex: i,
      time: point.time,
    }));
    sorted.sort((a, b) => a.time - b.time);
    return sorted;
  }, [allPoints]);

  const { positions, basePositions, colors, sizes, baseColors } = useMemo(() => {
    const positions = new Float32Array(allPoints.length * 3);
    const basePositions = new Float32Array(allPoints.length * 3);
    const colors = new Float32Array(allPoints.length * 3);
    const sizes = new Float32Array(allPoints.length);
    const baseColors: THREE.Color[] = [];

    allPoints.forEach((point, i) => {
      // Use PCA coordinates directly (already scaled appropriately)
      // Fall back to old transformation if coordinates seem to be in 0-1 range
      const isPcaCoords = Math.abs(point.x) > 2 || Math.abs(point.y) > 2 || Math.abs(point.z) > 2;

      if (isPcaCoords) {
        // Direct PCA coordinates (already in world space)
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
      } else {
        // Legacy transformation for old-style normalized coordinates
        const scale = settings.visualStyle === "galaxy" ? 8 : 5;

        if (settings.visualStyle === "galaxy") {
          const angle = point.x * Math.PI * 4;
          const radius = (0.3 + point.z * 0.7) * scale;
          positions[i * 3] = Math.cos(angle) * radius;
          positions[i * 3 + 1] = (point.y - 0.5) * scale * 0.5;
          positions[i * 3 + 2] = Math.sin(angle) * radius + (point.bandOffsetZ || 0) * scale;
        } else {
          positions[i * 3] = (point.x - 0.5) * scale * 2;
          positions[i * 3 + 1] = (point.y - 0.5) * scale;
          positions[i * 3 + 2] = (point.z - 0.5) * scale + (point.bandOffsetZ || 0) * scale;
        }
      }

      basePositions[i * 3] = positions[i * 3];
      basePositions[i * 3 + 1] = positions[i * 3 + 1];
      basePositions[i * 3 + 2] = positions[i * 3 + 2];

      const frequencyValue = point.color[0];
      const color = mapFrequencyToColor(frequencyValue);
      baseColors.push(color);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;

      sizes[i] = 0.08 + point.size * 0.15;
    });

    return { positions, basePositions, colors, sizes, baseColors };
  }, [allPoints, settings.visualStyle]);

  const { lineGeometry, sortedEdgeTimes } = useMemo(() => {
    if (settings.visualStyle !== "network") return { lineGeometry: null, sortedEdgeTimes: [] };

    interface EdgeData {
      fromGlobal: number;
      toGlobal: number;
      visibleAt: number;
    }
    const edges: EdgeData[] = [];
    let pointIndex = 0;

    data.verses.forEach(verse => {
      verse.edges.forEach(([from, to]) => {
        const fromGlobal = pointIndex + from;
        const toGlobal = pointIndex + to;
        if (fromGlobal < allPoints.length && toGlobal < allPoints.length) {
          const fromTime = allPoints[fromGlobal].time;
          const toTime = allPoints[toGlobal].time;
          edges.push({
            fromGlobal,
            toGlobal,
            visibleAt: Math.max(fromTime, toTime),
          });
        }
      });
      pointIndex += verse.points.length;
    });

    edges.sort((a, b) => a.visibleAt - b.visibleAt);

    const linePositions: number[] = [];
    const sortedEdgeTimes: number[] = [];

    edges.forEach(edge => {
      linePositions.push(
        positions[edge.fromGlobal * 3],
        positions[edge.fromGlobal * 3 + 1],
        positions[edge.fromGlobal * 3 + 2],
        positions[edge.toGlobal * 3],
        positions[edge.toGlobal * 3 + 1],
        positions[edge.toGlobal * 3 + 2]
      );
      sortedEdgeTimes.push(edge.visibleAt);
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    return { lineGeometry: geometry, sortedEdgeTimes };
  }, [data.verses, positions, allPoints, settings.visualStyle]);

  useEffect(() => {
    if (currentTime < 0.01) {
      maxRevealedRef.current = 0;
    }
  }, [currentTime]);

  useFrame((_, delta) => {
    if (!pointsRef.current) return;
    
    timeRef.current += delta;

    const colorsAttr = pointsRef.current.geometry.attributes.color;
    const sizesAttr = pointsRef.current.geometry.attributes.size;
    const positionsAttr = pointsRef.current.geometry.attributes.position;
    const colorsArray = colorsAttr.array as Float32Array;
    const sizesArray = sizesAttr.array as Float32Array;
    const positionsArray = positionsAttr.array as Float32Array;

    if (!currentSizesRef.current || currentSizesRef.current.length !== allPoints.length) {
      currentSizesRef.current = new Float32Array(allPoints.length);
      currentBrightnessRef.current = new Float32Array(allPoints.length);
      for (let i = 0; i < allPoints.length; i++) {
        currentSizesRef.current[i] = 0.08;
        currentBrightnessRef.current[i] = 1;
      }
    }

    const timeWindow = 0.03;
    const revealLeadTime = 0.08;
    const smoothingFactor = 0.15;
    const maxDelta = 0.2;
    const driftAmount = 0.015;
    const driftSpeed = 0.3;

    let visiblePointCount = allPoints.length;
    if (settings.progressiveReveal) {
      let count = 0;
      for (let i = 0; i < sortedPointIndices.length; i++) {
        if (sortedPointIndices[i].time <= currentTime + revealLeadTime) {
          count = i + 1;
        } else {
          break;
        }
      }
      visiblePointCount = Math.max(count, maxRevealedRef.current);
      maxRevealedRef.current = visiblePointCount;
    }

    let maxOnsetLow = 0, maxOnsetMid = 0, maxOnsetHigh = 0;
    for (let i = 0; i < allPoints.length; i++) {
      const point = allPoints[i];
      const dt = Math.abs(point.time - currentTime);
      if (dt < timeWindow) {
        maxOnsetLow = Math.max(maxOnsetLow, point.onsetLow || 0);
        maxOnsetMid = Math.max(maxOnsetMid, point.onsetMid || 0);
        maxOnsetHigh = Math.max(maxOnsetHigh, point.onsetHigh || 0);
      }
    }

    allPoints.forEach((point, i) => {
      const dt = Math.abs(point.time - currentTime);

      const bandOnset = point.band === 'low' ? (point.onsetLow || 0)
                      : point.band === 'mid' ? (point.onsetMid || 0)
                      : (point.onsetHigh || 0);
      const maxOnset = point.band === 'low' ? maxOnsetLow
                     : point.band === 'mid' ? maxOnsetMid
                     : maxOnsetHigh;

      let activation = 0;
      if (dt < timeWindow && maxOnset > 0.01) {
        const timeFalloff = 1 - (dt / timeWindow) * (dt / timeWindow);
        const normOnset = bandOnset / maxOnset;
        activation = Math.min(1, normOnset * timeFalloff);
      }

      const beatBoost = (point.beatStrength || 0) * activation;
      const complexityBoost = (point.complexity || 0) * 0.2;
      const targetIntensity = activation + beatBoost * 1.5;

      const isVisible = !settings.progressiveReveal || point.time <= currentTime + revealLeadTime;
      const alpha = isVisible ? 1 : 0;
      
      let r = 0, g = 0, b = 0;
      if (point.band === 'low')  { r = 1; g = 0.2; b = 0.2; }
      if (point.band === 'mid')  { r = 0.2; g = 1; b = 0.2; }
      if (point.band === 'high') { r = 0.2; g = 0.2; b = 1; }
      
      const targetBrightness = 0.3 + activation * 2.0 + complexityBoost;
      let currentBrightness = currentBrightnessRef.current![i];
      let brightnessDelta = (targetBrightness - currentBrightness) * smoothingFactor;
      brightnessDelta = Math.max(-maxDelta, Math.min(maxDelta, brightnessDelta));
      currentBrightness += brightnessDelta;
      currentBrightnessRef.current![i] = currentBrightness;
      
      colorsArray[i * 3] = Math.min(1, r * currentBrightness) * alpha;
      colorsArray[i * 3 + 1] = Math.min(1, g * currentBrightness) * alpha;
      colorsArray[i * 3 + 2] = Math.min(1, b * currentBrightness) * alpha;

      const baseSize = 0.06 + point.size * 0.1;
      const pulseSize = baseSize + activation * 0.8;
      const targetSize = isVisible ? pulseSize * (1 + beatBoost * 0.8) : 0;
      
      let currentSize = currentSizesRef.current![i];
      let sizeDelta = (targetSize - currentSize) * smoothingFactor;
      sizeDelta = Math.max(-maxDelta * 0.1, Math.min(maxDelta * 0.1, sizeDelta));
      currentSize += sizeDelta;
      currentSizesRef.current![i] = currentSize;
      sizesArray[i] = currentSize;
      
      if (isVisible) {
        const noiseX = simplex2D(i * 0.1, timeRef.current * driftSpeed) * driftAmount;
        const noiseY = simplex2D(i * 0.1 + 100, timeRef.current * driftSpeed) * driftAmount;
        const noiseZ = simplex2D(i * 0.1 + 200, timeRef.current * driftSpeed) * driftAmount;
        
        positionsArray[i * 3] = basePositions[i * 3] + noiseX;
        positionsArray[i * 3 + 1] = basePositions[i * 3 + 1] + noiseY;
        positionsArray[i * 3 + 2] = basePositions[i * 3 + 2] + noiseZ;
      }
    });

    colorsAttr.needsUpdate = true;
    sizesAttr.needsUpdate = true;
    positionsAttr.needsUpdate = true;

    if (linesRef.current && lineGeometry) {
      if (settings.progressiveReveal) {
        let visibleEdgeCount = 0;
        for (let i = 0; i < sortedEdgeTimes.length; i++) {
          if (sortedEdgeTimes[i] <= currentTime + revealLeadTime) {
            visibleEdgeCount = i + 1;
          } else {
            break;
          }
        }
        linesRef.current.geometry.setDrawRange(0, visibleEdgeCount * 2);
      } else {
        linesRef.current.geometry.setDrawRange(0, Infinity);
      }
    }
  });

  return (
    <group>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={allPoints.length}
            array={positions}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            count={allPoints.length}
            array={colors}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-size"
            count={allPoints.length}
            array={sizes}
            itemSize={1}
          />
        </bufferGeometry>
        <shaderMaterial
          vertexColors
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          vertexShader={`
            attribute float size;
            varying vec3 vColor;
            void main() {
              vColor = color;
              vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
              gl_PointSize = size * (300.0 / -mvPosition.z);
              gl_Position = projectionMatrix * mvPosition;
            }
          `}
          fragmentShader={`
            varying vec3 vColor;
            void main() {
              float dist = length(gl_PointCoord - vec2(0.5));
              if (dist > 0.5) discard;
              float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
              float glow = exp(-dist * 4.0) * 0.5;
              vec3 finalColor = vColor + vColor * glow;
              gl_FragColor = vec4(finalColor, alpha);
            }
          `}
        />
      </points>

      {lineGeometry && (
        <lineSegments ref={linesRef} geometry={lineGeometry}>
          <lineBasicMaterial
            color="#4488ff"
            transparent
            opacity={0.12}
            blending={THREE.AdditiveBlending}
          />
        </lineSegments>
      )}
    </group>
  );
}

function AutoRotate({ enabled }: { enabled: boolean }) {
  const { camera } = useThree();
  const targetSpeedRef = useRef(0);
  const currentSpeedRef = useRef(0);

  useFrame((_, delta) => {
    const targetSpeed = enabled ? 0.08 : 0;
    targetSpeedRef.current = targetSpeed;
    
    const speedDiff = targetSpeedRef.current - currentSpeedRef.current;
    currentSpeedRef.current += speedDiff * 0.02;
    
    if (Math.abs(currentSpeedRef.current) > 0.001) {
      camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), delta * currentSpeedRef.current);
      camera.lookAt(0, 0, 0);
    }
  });

  return null;
}

export function VisualizationCanvas({
  data,
  currentTime,
  settings,
}: VisualizationCanvasProps) {
  return (
    <div className="absolute inset-0 z-0" data-testid="canvas-visualization">
      <Canvas
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
        }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0A0A0A"]} />
        <fog attach="fog" args={["#0A0A0A", 10, 30]} />

        <PerspectiveCamera makeDefault position={[0, 2, 12]} fov={60} />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={3}
          maxDistance={30}
          enablePan={true}
          autoRotate={false}
        />

        <AutoRotate enabled={settings.autoRotate && !settings.isFullscreen} />

        <ambientLight intensity={0.1} />

        <PointsVisualization
          data={data}
          currentTime={currentTime}
          settings={settings}
        />
      </Canvas>
    </div>
  );
}
