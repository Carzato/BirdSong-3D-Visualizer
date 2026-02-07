import { z } from "zod";

export const audioFrameSchema = z.object({
  t: z.number(),
  pitch: z.number(),
  centroid: z.number(),
  bandwidth: z.number(),
  amplitude: z.number(),
  onsetStrength: z.number(),
  spectralFlux: z.number(),
  spectralFlatness: z.number(),
  beatStrength: z.number(),
  bandLow: z.number(),
  bandMid: z.number(),
  bandHigh: z.number(),
  onsetLow: z.number(),
  onsetMid: z.number(),
  onsetHigh: z.number(),
});

export const visualizationPointSchema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
  size: z.number(),
  color: z.tuple([z.number(), z.number(), z.number()]),
  time: z.number(),
  beatStrength: z.number(),
  complexity: z.number(),
  band: z.enum(['low', 'mid', 'high']),
  bandOffsetZ: z.number(),
  onsetLow: z.number(),
  onsetMid: z.number(),
  onsetHigh: z.number(),
});

export const verseSchema = z.object({
  id: z.number(),
  name: z.string(),
  start: z.number(),
  end: z.number(),
  points: z.array(visualizationPointSchema),
  edges: z.array(z.tuple([z.number(), z.number()])),
});

export const visualizationDataSchema = z.object({
  audioUrl: z.string(),
  duration: z.number(),
  sampleRate: z.number(),
  verses: z.array(verseSchema),
});

export const audioUploadResponseSchema = z.object({
  success: z.boolean(),
  data: visualizationDataSchema.optional(),
  error: z.string().optional(),
});

export const visualStyleSchema = z.enum(["network", "galaxy", "ribbons"]);

export type AudioFrame = z.infer<typeof audioFrameSchema>;
export type VisualizationPoint = z.infer<typeof visualizationPointSchema>;
export type Verse = z.infer<typeof verseSchema>;
export type VisualizationData = z.infer<typeof visualizationDataSchema>;
export type AudioUploadResponse = z.infer<typeof audioUploadResponseSchema>;
export type VisualStyle = z.infer<typeof visualStyleSchema>;

export interface VisualizationSettings {
  visualStyle: VisualStyle;
  autoRotate: boolean;
  loopPlayback: boolean;
  showDebug: boolean;
  isFullscreen: boolean;
  progressiveReveal: boolean;
}
