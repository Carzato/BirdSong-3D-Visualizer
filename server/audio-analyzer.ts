import fs from "fs";
import path from "path";
import ffmpeg from "fluent-ffmpeg";
import { randomUUID } from "crypto";
import os from "os";

interface AudioFrame {
  t: number;
  pitch: number;
  centroid: number;
  bandwidth: number;
  amplitude: number;
  onsetStrength: number;
  spectralFlux: number;
  spectralFlatness: number;
  beatStrength: number;
  bandLow: number;
  bandMid: number;
  bandHigh: number;
  onsetLow: number;
  onsetMid: number;
  onsetHigh: number;
}

interface VisualizationPoint {
  x: number;
  y: number;
  z: number;
  size: number;
  color: [number, number, number];
  time: number;
  beatStrength: number;
  complexity: number;
  band: 'low' | 'mid' | 'high';
  bandOffsetZ: number;
  onset: number;
}

interface Verse {
  id: number;
  name: string;
  start: number;
  end: number;
  points: VisualizationPoint[];
  edges: [number, number][];
}

interface AnalysisResult {
  duration: number;
  sampleRate: number;
  verses: Verse[];
}

function parseWavHeader(buffer: Buffer): { sampleRate: number; channels: number; bitsPerSample: number; dataOffset: number; dataLength: number } | null {
  if (buffer.length < 44) return null;
  
  const riff = buffer.toString("ascii", 0, 4);
  const wave = buffer.toString("ascii", 8, 12);
  
  if (riff !== "RIFF" || wave !== "WAVE") return null;
  
  let offset = 12;
  let sampleRate = 44100;
  let channels = 2;
  let bitsPerSample = 16;
  let dataOffset = 44;
  let dataLength = 0;
  
  while (offset < buffer.length - 8) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    
    if (chunkId === "fmt ") {
      channels = buffer.readUInt16LE(offset + 10);
      sampleRate = buffer.readUInt32LE(offset + 12);
      bitsPerSample = buffer.readUInt16LE(offset + 22);
    } else if (chunkId === "data") {
      dataOffset = offset + 8;
      dataLength = chunkSize;
      break;
    }
    
    offset += 8 + chunkSize;
  }
  
  return { sampleRate, channels, bitsPerSample, dataOffset, dataLength };
}

function extractPcmSamples(buffer: Buffer, header: { sampleRate: number; channels: number; bitsPerSample: number; dataOffset: number; dataLength: number }): Float32Array {
  const { dataOffset, dataLength, bitsPerSample, channels } = header;
  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataLength / (bytesPerSample * channels));
  const samples = new Float32Array(numSamples);
  
  for (let i = 0; i < numSamples; i++) {
    const offset = dataOffset + i * bytesPerSample * channels;
    if (offset + bytesPerSample > buffer.length) break;
    
    let sample: number;
    if (bitsPerSample === 16) {
      sample = buffer.readInt16LE(offset) / 32768;
    } else if (bitsPerSample === 8) {
      sample = (buffer.readUInt8(offset) - 128) / 128;
    } else if (bitsPerSample === 32) {
      sample = buffer.readInt32LE(offset) / 2147483648;
    } else {
      sample = buffer.readInt16LE(offset) / 32768;
    }
    
    samples[i] = sample;
  }
  
  return samples;
}

function computeRMS(samples: Float32Array, start: number, length: number): number {
  let sum = 0;
  const end = Math.min(start + length, samples.length);
  for (let i = start; i < end; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / (end - start));
}

function computeZeroCrossingRate(samples: Float32Array, start: number, length: number): number {
  let crossings = 0;
  const end = Math.min(start + length, samples.length);
  for (let i = start + 1; i < end; i++) {
    if ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0)) {
      crossings++;
    }
  }
  return crossings / (end - start);
}

function estimatePitchFromZCR(zcr: number, sampleRate: number): number {
  const estimatedFreq = (zcr * sampleRate) / 2;
  return Math.max(80, Math.min(8000, estimatedFreq));
}

function computeFFT(samples: Float32Array, start: number, frameSize: number): Float32Array {
  const fft = new Float32Array(frameSize / 2);
  
  for (let k = 0; k < frameSize / 2; k++) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < frameSize && start + n < samples.length; n++) {
      const angle = (2 * Math.PI * k * n) / frameSize;
      real += samples[start + n] * Math.cos(angle);
      imag -= samples[start + n] * Math.sin(angle);
    }
    fft[k] = Math.sqrt(real * real + imag * imag);
  }
  
  return fft;
}

function computeSpectralFeatures(fft: Float32Array, frameSize: number, sampleRate: number): { centroid: number; bandwidth: number } {
  const freqBinWidth = sampleRate / frameSize;
  
  let sumMag = 0;
  let sumFreqMag = 0;
  for (let k = 1; k < fft.length; k++) {
    const freq = k * freqBinWidth;
    sumMag += fft[k];
    sumFreqMag += freq * fft[k];
  }
  
  const centroid = sumMag > 0 ? sumFreqMag / sumMag : 1000;
  
  let sumBandwidth = 0;
  for (let k = 1; k < fft.length; k++) {
    const freq = k * freqBinWidth;
    sumBandwidth += fft[k] * Math.pow(freq - centroid, 2);
  }
  const bandwidth = sumMag > 0 ? Math.sqrt(sumBandwidth / sumMag) : 500;
  
  return { centroid, bandwidth };
}

function computeSpectralFlux(currentFFT: Float32Array, prevFFT: Float32Array | null): number {
  if (!prevFFT) return 0;
  
  let flux = 0;
  for (let k = 0; k < currentFFT.length; k++) {
    const diff = currentFFT[k] - prevFFT[k];
    if (diff > 0) {
      flux += diff * diff;
    }
  }
  return Math.sqrt(flux);
}

function computeSpectralFlatness(fft: Float32Array): number {
  const epsilon = 1e-10;
  let logSum = 0;
  let sum = 0;
  let count = 0;
  
  for (let k = 1; k < fft.length; k++) {
    const mag = Math.max(fft[k], epsilon);
    logSum += Math.log(mag);
    sum += mag;
    count++;
  }
  
  if (count === 0 || sum === 0) return 0;
  
  const geometricMean = Math.exp(logSum / count);
  const arithmeticMean = sum / count;
  
  return geometricMean / arithmeticMean;
}

function computeOnsetStrength(prevRMS: number, currentRMS: number): number {
  const diff = currentRMS - prevRMS;
  return diff > 0 ? Math.min(1, diff * 10) : 0;
}

function computeBandEnergies(
  fft: Float32Array,
  fftSize: number,
  sampleRate: number
): { low: number; mid: number; high: number } {
  const binFreq = (bin: number) => bin * sampleRate / fftSize;

  let low = 0, mid = 0, high = 0;

  for (let bin = 0; bin < fft.length; bin++) {
    const freq = binFreq(bin);
    const mag = fft[bin];

    if (freq >= 20 && freq < 250) {
      low += mag;
    } else if (freq >= 250 && freq < 2000) {
      mid += mag;
    } else if (freq >= 2000 && freq < 8000) {
      high += mag;
    }
  }

  return { low, mid, high };
}

function computeBandOnsets(frames: AudioFrame[]): void {
  if (frames.length < 2) return;
  
  let prevLow = frames[0].bandLow;
  let prevMid = frames[0].bandMid;
  let prevHigh = frames[0].bandHigh;

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];

    const dLow = Math.max(0, f.bandLow - prevLow);
    const dMid = Math.max(0, f.bandMid - prevMid);
    const dHigh = Math.max(0, f.bandHigh - prevHigh);

    f.onsetLow = dLow;
    f.onsetMid = dMid;
    f.onsetHigh = dHigh;

    prevLow = f.bandLow;
    prevMid = f.bandMid;
    prevHigh = f.bandHigh;
  }
  
  const maxOnsetLow = Math.max(...frames.map(f => f.onsetLow), 0.001);
  const maxOnsetMid = Math.max(...frames.map(f => f.onsetMid), 0.001);
  const maxOnsetHigh = Math.max(...frames.map(f => f.onsetHigh), 0.001);
  
  for (const f of frames) {
    f.onsetLow /= maxOnsetLow;
    f.onsetMid /= maxOnsetMid;
    f.onsetHigh /= maxOnsetHigh;
  }
}

function applyTemporalSmoothing(frames: AudioFrame[]): void {
  if (frames.length < 3) return;
  
  const alpha = 0.5;
  
  let smoothPitch = frames[0].pitch;
  let smoothCentroid = frames[0].centroid;
  let smoothAmplitude = frames[0].amplitude;
  let smoothBandwidth = frames[0].bandwidth;
  let smoothBandLow = frames[0].bandLow;
  let smoothBandMid = frames[0].bandMid;
  let smoothBandHigh = frames[0].bandHigh;
  
  for (let i = 0; i < frames.length; i++) {
    smoothPitch = alpha * frames[i].pitch + (1 - alpha) * smoothPitch;
    smoothCentroid = alpha * frames[i].centroid + (1 - alpha) * smoothCentroid;
    smoothAmplitude = alpha * frames[i].amplitude + (1 - alpha) * smoothAmplitude;
    smoothBandwidth = alpha * frames[i].bandwidth + (1 - alpha) * smoothBandwidth;
    smoothBandLow = alpha * frames[i].bandLow + (1 - alpha) * smoothBandLow;
    smoothBandMid = alpha * frames[i].bandMid + (1 - alpha) * smoothBandMid;
    smoothBandHigh = alpha * frames[i].bandHigh + (1 - alpha) * smoothBandHigh;
    
    frames[i].pitch = smoothPitch;
    frames[i].centroid = smoothCentroid;
    frames[i].amplitude = smoothAmplitude;
    frames[i].bandwidth = smoothBandwidth;
    frames[i].bandLow = smoothBandLow;
    frames[i].bandMid = smoothBandMid;
    frames[i].bandHigh = smoothBandHigh;
  }
}

function detectBeats(frames: AudioFrame[], sampleRate: number): void {
  if (frames.length < 10) return;
  
  const fluxValues = frames.map(f => f.spectralFlux);
  const maxFlux = Math.max(...fluxValues);
  if (maxFlux === 0) return;
  
  const normalizedFlux = fluxValues.map(f => f / maxFlux);
  
  const windowSize = Math.min(10, Math.floor(frames.length / 4));
  const threshold: number[] = [];
  
  for (let i = 0; i < frames.length; i++) {
    const start = Math.max(0, i - windowSize);
    const end = Math.min(frames.length, i + windowSize + 1);
    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += normalizedFlux[j];
    }
    const mean = sum / (end - start);
    threshold[i] = mean * 1.3 + 0.1;
  }
  
  for (let i = 1; i < frames.length - 1; i++) {
    const isPeak = normalizedFlux[i] > normalizedFlux[i - 1] && 
                   normalizedFlux[i] >= normalizedFlux[i + 1];
    const aboveThreshold = normalizedFlux[i] > threshold[i];
    
    if (isPeak && aboveThreshold) {
      frames[i].beatStrength = Math.min(1, normalizedFlux[i] * 1.5);
    }
  }
  
  const beatStrengths = frames.map(f => f.beatStrength);
  const maxBeatStrength = Math.max(...beatStrengths);
  if (maxBeatStrength > 0) {
    for (const frame of frames) {
      frame.beatStrength = frame.beatStrength / maxBeatStrength;
    }
  }
}

function extractAudioFrames(samples: Float32Array, sampleRate: number): AudioFrame[] {
  const frameSize = 2048;
  const hopSize = 512;
  const fftSize = 512;
  const numFrames = Math.floor((samples.length - frameSize) / hopSize);
  const frames: AudioFrame[] = [];
  
  let prevRMS = 0;
  let prevFFT: Float32Array | null = null;
  
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const t = start / sampleRate;
    
    const amplitude = computeRMS(samples, start, frameSize);
    const zcr = computeZeroCrossingRate(samples, start, frameSize);
    const pitch = estimatePitchFromZCR(zcr, sampleRate);
    
    const fft = computeFFT(samples, start, fftSize);
    const { centroid, bandwidth } = computeSpectralFeatures(fft, fftSize, sampleRate);
    const spectralFlux = computeSpectralFlux(fft, prevFFT);
    const spectralFlatness = computeSpectralFlatness(fft);
    const onsetStrength = computeOnsetStrength(prevRMS, amplitude);
    const bandEnergies = computeBandEnergies(fft, fftSize, sampleRate);
    
    frames.push({
      t,
      pitch,
      centroid,
      bandwidth,
      amplitude,
      onsetStrength,
      spectralFlux,
      spectralFlatness,
      beatStrength: 0,
      bandLow: bandEnergies.low,
      bandMid: bandEnergies.mid,
      bandHigh: bandEnergies.high,
      onsetLow: 0,
      onsetMid: 0,
      onsetHigh: 0,
    });
    
    prevRMS = amplitude;
    prevFFT = fft;
  }
  
  applyTemporalSmoothing(frames);
  computeBandOnsets(frames);
  detectBeats(frames, sampleRate);
  
  return frames;
}

function segmentIntoVerses(frames: AudioFrame[], duration: number): { start: number; end: number; frames: AudioFrame[] }[] {
  const verses: { start: number; end: number; frames: AudioFrame[] }[] = [];
  
  if (frames.length === 0) return verses;

  const maxAmp = Math.max(...frames.map(f => f.amplitude));
  const silenceThreshold = maxAmp * 0.1;
  
  let verseStart = 0;
  let verseFrames: AudioFrame[] = [];
  let inSilence = false;
  let silenceStart = 0;
  const minSilenceDuration = 0.2;
  const minVerseDuration = 0.3;

  for (const frame of frames) {
    const isSilent = frame.amplitude < silenceThreshold;

    if (isSilent && !inSilence) {
      inSilence = true;
      silenceStart = frame.t;
    } else if (!isSilent && inSilence) {
      const silenceDuration = frame.t - silenceStart;
      if (silenceDuration > minSilenceDuration && verseFrames.length > 0) {
        const verseDuration = silenceStart - verseStart;
        if (verseDuration >= minVerseDuration) {
          verses.push({
            start: verseStart,
            end: silenceStart,
            frames: [...verseFrames],
          });
        }
        verseStart = frame.t;
        verseFrames = [];
      }
      inSilence = false;
    }

    if (!isSilent) {
      verseFrames.push(frame);
    }
  }

  if (verseFrames.length > 0) {
    verses.push({
      start: verseStart,
      end: duration,
      frames: verseFrames,
    });
  }

  if (verses.length === 0 && frames.length > 0) {
    const segmentDuration = duration / Math.min(4, Math.ceil(duration / 5));
    const numSegments = Math.ceil(duration / segmentDuration);
    for (let i = 0; i < numSegments; i++) {
      const start = i * segmentDuration;
      const end = Math.min((i + 1) * segmentDuration, duration);
      const segmentFrames = frames.filter(f => f.t >= start && f.t < end);
      if (segmentFrames.length > 0) {
        verses.push({ start, end, frames: segmentFrames });
      }
    }
  }

  return verses;
}

function normalizeValue(value: number, min: number, max: number): number {
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function findKNearestNeighbors(points: VisualizationPoint[], k: number): [number, number][] {
  const edges: [number, number][] = [];
  const edgeSet = new Set<string>();

  for (let i = 0; i < points.length; i++) {
    const distances: { index: number; dist: number }[] = [];

    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;

      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      const dz = points[i].z - points[j].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      distances.push({ index: j, dist });
    }

    distances.sort((a, b) => a.dist - b.dist);

    for (let n = 0; n < Math.min(k, distances.length); n++) {
      const j = distances[n].index;
      const edgeKey = i < j ? `${i}-${j}` : `${j}-${i}`;

      if (!edgeSet.has(edgeKey)) {
        edgeSet.add(edgeKey);
        edges.push([i, j]);
      }
    }
  }

  return edges;
}

function pickDominantBand(frame: AudioFrame): { band: 'low' | 'mid' | 'high'; onset: number } {
  const { bandLow, bandMid, bandHigh } = frame;

  if (bandMid >= bandLow && bandMid >= bandHigh) {
    return { band: 'mid', onset: frame.onsetMid };
  } else if (bandLow >= bandMid && bandLow >= bandHigh) {
    return { band: 'low', onset: frame.onsetLow };
  } else {
    return { band: 'high', onset: frame.onsetHigh };
  }
}

function mapFramesToVisualization(
  verseSegments: { start: number; end: number; frames: AudioFrame[] }[],
  duration: number
): Verse[] {
  const allFrames = verseSegments.flatMap(v => v.frames);
  
  if (allFrames.length === 0) {
    return [];
  }

  const pitchValues = allFrames.map(f => Math.log2(Math.max(1, f.pitch)));
  const pitchMin = Math.min(...pitchValues);
  const pitchMax = Math.max(...pitchValues);
  const centroidMin = Math.min(...allFrames.map(f => f.centroid));
  const centroidMax = Math.max(...allFrames.map(f => f.centroid));
  const bandwidthMin = Math.min(...allFrames.map(f => f.bandwidth));
  const bandwidthMax = Math.max(...allFrames.map(f => f.bandwidth));
  const amplitudeMax = Math.max(...allFrames.map(f => f.amplitude));

  return verseSegments.map((segment, verseIndex) => {
    const verseDuration = segment.end - segment.start;

    const maxPoints = 400;
    const step = Math.max(1, Math.floor(segment.frames.length / maxPoints));
    const sampledFrames = segment.frames.filter((_, i) => i % step === 0);

    const points: VisualizationPoint[] = sampledFrames.map(frame => {
      const normalizedTime = verseDuration > 0 ? (frame.t - segment.start) / verseDuration : 0;

      const normalizedPitch = normalizeValue(Math.log2(Math.max(1, frame.pitch)), pitchMin, pitchMax);
      const normalizedCentroid = normalizeValue(frame.centroid, centroidMin, centroidMax);
      const normalizedBandwidth = normalizeValue(frame.bandwidth, bandwidthMin, bandwidthMax);
      const normalizedAmplitude = amplitudeMax > 0 ? frame.amplitude / amplitudeMax : 0;

      const hue = normalizedCentroid;
      const saturation = 0.7 + normalizedBandwidth * 0.3;
      const lightness = 0.4 + normalizedAmplitude * 0.4;
      
      const complexity = (1 - frame.spectralFlatness) * normalizedBandwidth;

      const { band } = pickDominantBand(frame);
      
      let bandOffsetZ = 0;
      if (band === 'low') bandOffsetZ = -0.8;
      if (band === 'mid') bandOffsetZ = 0;
      if (band === 'high') bandOffsetZ = 0.8;

      return {
        x: normalizedTime,
        y: normalizedPitch,
        z: normalizedCentroid,
        size: 0.3 + normalizedAmplitude * 0.7,
        color: [hue, saturation, lightness] as [number, number, number],
        time: frame.t,
        beatStrength: frame.beatStrength,
        complexity,
        band,
        bandOffsetZ,
        onsetLow: frame.onsetLow,
        onsetMid: frame.onsetMid,
        onsetHigh: frame.onsetHigh,
      };
    });

    const edges = points.length > 1 ? findKNearestNeighbors(points, 3) : [];

    return {
      id: verseIndex,
      name: `Phrase ${verseIndex + 1}`,
      start: segment.start,
      end: segment.end,
      points,
      edges,
    };
  });
}

async function transcodeToWav(inputPath: string): Promise<string> {
  const tempWavPath = path.join(os.tmpdir(), `${randomUUID()}.wav`);
  
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .toFormat("wav")
      .audioCodec("pcm_s16le")
      .audioChannels(1)
      .audioFrequency(44100)
      .on("error", (err) => {
        reject(new Error(`Failed to transcode audio: ${err.message}`));
      })
      .on("end", () => {
        resolve(tempWavPath);
      })
      .save(tempWavPath);
  });
}

export async function analyzeAudioFile(filePath: string): Promise<AnalysisResult> {
  const ext = path.extname(filePath).toLowerCase();
  const supportedFormats = [".wav", ".mp3", ".m4a"];
  
  if (!supportedFormats.includes(ext)) {
    throw new Error("Unsupported audio format. Please upload WAV, MP3, or M4A files.");
  }
  
  let wavFilePath = filePath;
  let needsCleanup = false;
  
  if (ext !== ".wav") {
    wavFilePath = await transcodeToWav(filePath);
    needsCleanup = true;
  }
  
  try {
    const buffer = fs.readFileSync(wavFilePath);
    
    const header = parseWavHeader(buffer);
    if (!header) {
      throw new Error("Failed to parse audio file. Please ensure it's a valid audio file.");
    }
    
    const sampleRate = header.sampleRate;
    const samples = extractPcmSamples(buffer, header);
    
    if (samples.length === 0) {
      throw new Error("Could not extract audio samples from the file.");
    }
    
    const duration = samples.length / sampleRate;
    
    const frames = extractAudioFrames(samples, sampleRate);
    
    const verseSegments = segmentIntoVerses(frames, duration);
    
    const verses = mapFramesToVisualization(verseSegments, duration);
    
    return {
      duration,
      sampleRate,
      verses,
    };
  } finally {
    if (needsCleanup && fs.existsSync(wavFilePath)) {
      fs.unlinkSync(wavFilePath);
    }
  }
}

// Keep this for future MP3/M4A support
function _generateFallbackSamples(buffer: Buffer, sampleRate: number): Float32Array {
  const fileSize = buffer.length;
  const estimatedDuration = Math.min(60, Math.max(5, fileSize / (44100 * 2)));
  const numSamples = Math.floor(estimatedDuration * sampleRate);
  const samples = new Float32Array(numSamples);
  
  for (let i = 0; i < numSamples; i++) {
    const bufferIndex = Math.floor((i / numSamples) * buffer.length);
    const byte1 = buffer[bufferIndex] || 0;
    const byte2 = buffer[Math.min(bufferIndex + 1, buffer.length - 1)] || 0;
    const value = ((byte1 << 8) | byte2) / 65535;
    samples[i] = (value - 0.5) * 2;
  }
  
  return samples;
}

export async function getAudioDuration(filePath: string): Promise<number> {
  const ext = path.extname(filePath).toLowerCase();
  
  if (ext === ".wav") {
    const buffer = fs.readFileSync(filePath);
    const header = parseWavHeader(buffer);
    if (header) {
      const numSamples = header.dataLength / (header.bitsPerSample / 8) / header.channels;
      return numSamples / header.sampleRate;
    }
  }
  
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) {
        reject(new Error(`Failed to get audio duration: ${err.message}`));
        return;
      }
      const duration = metadata.format.duration || 0;
      resolve(duration);
    });
  });
}
