import { randomUUID } from "crypto";

export interface AudioAnalysis {
  id: string;
  filename: string;
  duration: number;
  sampleRate: number;
  createdAt: Date;
}

export interface IStorage {
  saveAudioAnalysis(analysis: Omit<AudioAnalysis, "id" | "createdAt">): Promise<AudioAnalysis>;
  getAudioAnalysis(id: string): Promise<AudioAnalysis | undefined>;
}

export class MemStorage implements IStorage {
  private analyses: Map<string, AudioAnalysis>;

  constructor() {
    this.analyses = new Map();
  }

  async saveAudioAnalysis(analysis: Omit<AudioAnalysis, "id" | "createdAt">): Promise<AudioAnalysis> {
    const id = randomUUID();
    const fullAnalysis: AudioAnalysis = {
      ...analysis,
      id,
      createdAt: new Date(),
    };
    this.analyses.set(id, fullAnalysis);
    return fullAnalysis;
  }

  async getAudioAnalysis(id: string): Promise<AudioAnalysis | undefined> {
    return this.analyses.get(id);
  }
}

export const storage = new MemStorage();
