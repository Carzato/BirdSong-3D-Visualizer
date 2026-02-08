import { z } from "zod";

export const audioFrameSchema = z.object({
  t: z.number(),
  mfccs: z.array(z.number()),
  pcaCoordinates: z.object({
    x: z.number(),
    y: z.number(),
    z: z.number(),
  }).optional(),
  frequency: z.number().optional(),
  amplitude: z.number().optional(),
});

export const visualizationPointSchema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
  size: z.number(),
  color: z.tuple([z.number(), z.number(), z.number()]),
  time: z.number(),
  mfccs: z.array(z.number()),
  frequency: z.number().optional(),
  beatStrength: z.number().optional(),
  complexity: z.number().optional(),
  band: z.enum(['low', 'mid', 'high']).optional(),
  bandOffsetZ: z.number().optional(),
  onsetLow: z.number().optional(),
  onsetMid: z.number().optional(),
  onsetHigh: z.number().optional(),
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
