import fs from "fs";
import path from "path";
import ffmpeg from "fluent-ffmpeg";
import { randomUUID } from "crypto";
import os from "os";
import { Matrix, EigenvalueDecomposition } from "ml-matrix";

interface AudioFrame {
  t: number;
  mfccs: number[];
  pcaCoordinates?: {
    x: number;
    y: number;
    z: number;
  };
  frequency?: number;
  amplitude?: number;
}

interface VisualizationPoint {
  x: number;
  y: number;
  z: number;
  size: number;
  color: [number, number, number];
  time: number;
  mfccs: number[];
  frequency?: number;
  beatStrength?: number;
  complexity?: number;
  band?: 'low' | 'mid' | 'high';
  bandOffsetZ?: number;
  onsetLow?: number;
  onsetMid?: number;
  onsetHigh?: number;
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

// Mel scale conversion functions
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

// Create Mel filterbank
function createMelFilterbank(
  numFilters: number,
  fftSize: number,
  sampleRate: number,
  lowFreq: number = 0,
  highFreq?: number
): number[][] {
  highFreq = highFreq || sampleRate / 2;

  const lowMel = hzToMel(lowFreq);
  const highMel = hzToMel(highFreq);
  const melPoints = Array.from(
    { length: numFilters + 2 },
    (_, i) => lowMel + (i * (highMel - lowMel)) / (numFilters + 1)
  );

  const hzPoints = melPoints.map(melToHz);
  const binPoints = hzPoints.map(hz => Math.floor((fftSize + 1) * hz / sampleRate));

  const filterbank: number[][] = [];
  for (let i = 1; i <= numFilters; i++) {
    const filter = new Array(Math.floor(fftSize / 2) + 1).fill(0);

    const left = binPoints[i - 1];
    const center = binPoints[i];
    const right = binPoints[i + 1];

    for (let j = left; j < center; j++) {
      filter[j] = (j - left) / (center - left);
    }
    for (let j = center; j < right; j++) {
      filter[j] = (right - j) / (right - center);
    }

    filterbank.push(filter);
  }

  return filterbank;
}

// Discrete Cosine Transform
function dct(input: number[], numCoefficients: number): number[] {
  const N = input.length;
  const output: number[] = [];

  for (let k = 0; k < numCoefficients; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n] * Math.cos((Math.PI * k * (n + 0.5)) / N);
    }
    output.push(sum);
  }

  return output;
}

// Extract MFCCs from audio frame
function extractMFCCs(
  samples: Float32Array,
  start: number,
  frameSize: number,
  sampleRate: number,
  numMFCCs: number = 40
): number[] {
  // Apply FFT
  const fft = computeFFT(samples, start, frameSize);

  // Create mel filterbank (26 filters is standard)
  const numFilters = 26;
  const filterbank = createMelFilterbank(numFilters, frameSize, sampleRate);

  // Apply filterbank to power spectrum
  const melEnergies: number[] = [];
  for (const filter of filterbank) {
    let energy = 0;
    for (let i = 0; i < fft.length; i++) {
      energy += (fft[i] * fft[i]) * filter[i];
    }
    // Take log of energy (add small epsilon to avoid log(0))
    melEnergies.push(Math.log(energy + 1e-10));
  }

  // Apply DCT to get MFCCs
  const mfccs = dct(melEnergies, numMFCCs);

  return mfccs;
}

// Calculate dominant frequency from FFT
function calculateDominantFrequency(fft: Float32Array, sampleRate: number, fftSize: number): number {
  let maxMag = 0;
  let maxBin = 0;

  for (let i = 1; i < fft.length; i++) {
    if (fft[i] > maxMag) {
      maxMag = fft[i];
      maxBin = i;
    }
  }

  return (maxBin * sampleRate) / fftSize;
}

function extractAudioFrames(samples: Float32Array, sampleRate: number): AudioFrame[] {
  const frameSize = 512;
  const hopSize = 256;  // 50% overlap
  const numFrames = Math.floor((samples.length - frameSize) / hopSize);
  const frames: AudioFrame[] = [];

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const t = start / sampleRate;

    // Extract 40 MFCCs
    const mfccs = extractMFCCs(samples, start, frameSize, sampleRate, 40);

    // Calculate amplitude for point sizing
    const amplitude = computeRMS(samples, start, frameSize);

    // Calculate dominant frequency for coloring
    const fft = computeFFT(samples, start, frameSize);
    const frequency = calculateDominantFrequency(fft, sampleRate, frameSize);

    frames.push({
      t,
      mfccs,
      amplitude,
      frequency,
    });
  }

  return frames;
}

// Expected MFCC dimension (must match extractMFCCs output length)
const PCA_MFCC_DIM = 40;

// Fallback 3D from first 3 MFCCs when PCA cannot be applied
function fallbackPcaCoordinates(frames: AudioFrame[]): AudioFrame[] {
  return frames.map((frame) => ({
    ...frame,
    pcaCoordinates: {
      x: (frame.mfccs[0] ?? 0) * 0.1,
      y: (frame.mfccs[1] ?? 0) * 0.1,
      z: (frame.mfccs[2] ?? 0) * 0.1,
    },
  }));
}

// Apply PCA to reduce MFCC dimensions from 40 to 3
function applyPCA(frames: AudioFrame[]): AudioFrame[] {
  if (frames.length === 0) return frames;

  try {
    // Normalize every frame to exactly PCA_MFCC_DIM coefficients (pad or truncate)
    // to avoid "Matrices dimensions must be equal" from jagged or inconsistent lengths
    const mfccData = frames.map((frame) => {
      const m = frame.mfccs;
      if (m.length === PCA_MFCC_DIM) return [...m];
      if (m.length > PCA_MFCC_DIM) return m.slice(0, PCA_MFCC_DIM);
      const row = [...m];
      while (row.length < PCA_MFCC_DIM) row.push(0);
      return row;
    });

    const mfccMatrix = new Matrix(mfccData);
    const numCols = mfccMatrix.columns;
    if (numCols !== PCA_MFCC_DIM) return fallbackPcaCoordinates(frames);
    if (frames.length < 2) return fallbackPcaCoordinates(frames);

    // Center the data (subtract mean from each column)
    const means = mfccMatrix.mean('column');
    const centered = mfccMatrix.clone();
    for (let col = 0; col < centered.columns; col++) {
      for (let row = 0; row < centered.rows; row++) {
        centered.set(row, col, centered.get(row, col) - means[col]);
      }
    }

    // Compute covariance matrix (symmetric: numCols x numCols)
    const transposed = centered.transpose();
    const covariance = transposed.mmul(centered).div(frames.length - 1);

    // Eigenvalue decomposition (covariance is symmetric by construction)
    const evd = new EigenvalueDecomposition(covariance, { assumeSymmetric: true });
    const eigenvectors = evd.eigenvectorMatrix;
    const eigenvalues = evd.realEigenvalues;

    if (eigenvectors.rows !== numCols || eigenvectors.columns !== numCols) {
      return fallbackPcaCoordinates(frames);
    }

    // Sort eigenvectors by eigenvalues (descending)
    const sorted = eigenvalues
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val);

    // Get top 3 principal components
    const numComponents = 3;
    const principalComponents = new Matrix(eigenvectors.rows, numComponents);
    for (let i = 0; i < numComponents; i++) {
      const colIdx = sorted[i].idx;
      for (let row = 0; row < eigenvectors.rows; row++) {
        principalComponents.set(row, i, eigenvectors.get(row, colIdx));
      }
    }

    // Project: centered (frames x numCols) * principalComponents (numCols x 3)
    const projected = centered.mmul(principalComponents);

    // Normalize coordinates to reasonable range for visualization
    const maxRange = Math.max(
      Math.max(...projected.getColumn(0).map(Math.abs)),
      Math.max(...projected.getColumn(1).map(Math.abs)),
      Math.max(...projected.getColumn(2).map(Math.abs))
    );

    const scale = maxRange > 0 ? 5 / maxRange : 1;

    return frames.map((frame, i) => ({
      ...frame,
      pcaCoordinates: {
        x: projected.get(i, 0) * scale,
        y: projected.get(i, 1) * scale,
        z: projected.get(i, 2) * scale,
      }
    }));
  } catch {
    // On any error (e.g. "Matrices dimensions must be equal"), use fallback 3D
    return fallbackPcaCoordinates(frames);
  }
}

function segmentIntoVerses(frames: AudioFrame[], duration: number): { start: number; end: number; frames: AudioFrame[] }[] {
  const verses: { start: number; end: number; frames: AudioFrame[] }[] = [];
  
  if (frames.length === 0) return verses;

  const maxAmp = Math.max(...frames.map(f => f.amplitude || 0));
  const silenceThreshold = maxAmp * 0.1;

  let verseStart = 0;
  let verseFrames: AudioFrame[] = [];
  let inSilence = false;
  let silenceStart = 0;
  const minSilenceDuration = 0.2;
  const minVerseDuration = 0.3;

  for (const frame of frames) {
    const isSilent = (frame.amplitude || 0) < silenceThreshold;

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

// Map frequency to hue for color coding (red/orange/yellow gradient)
function mapFrequencyToHue(frequency: number): number {
  // Map frequency range 0-10kHz to hue 0-60° (red→orange→yellow)
  const normalized = Math.min(frequency / 10000, 1);
  return normalized * 60;
}

function mapFramesToVisualization(
  verseSegments: { start: number; end: number; frames: AudioFrame[] }[],
  duration: number
): Verse[] {
  const allFrames = verseSegments.flatMap(v => v.frames);

  if (allFrames.length === 0) {
    return [];
  }

  const amplitudeMax = Math.max(...allFrames.map(f => f.amplitude || 0));

  return verseSegments.map((segment, verseIndex) => {
    const maxPoints = 400;
    const step = Math.max(1, Math.floor(segment.frames.length / maxPoints));
    const sampledFrames = segment.frames.filter((_, i) => i % step === 0);

    const points: VisualizationPoint[] = sampledFrames.map(frame => {
      // Use PCA coordinates directly
      const { x, y, z } = frame.pcaCoordinates || { x: 0, y: 0, z: 0 };

      // Frequency-based color (match video: red/orange/yellow)
      const hue = mapFrequencyToHue(frame.frequency || 0);
      const saturation = 0.8;
      const lightness = 0.6;

      // Size based on amplitude
      const normalizedAmplitude = amplitudeMax > 0 ? (frame.amplitude || 0) / amplitudeMax : 0.5;
      const size = 0.05 + normalizedAmplitude * 0.15;

      return {
        x,
        y,
        z,
        size,
        color: [hue, saturation, lightness] as [number, number, number],
        time: frame.t,
        mfccs: frame.mfccs,
        frequency: frame.frequency,
      };
    });

    // Compute k-nearest neighbors based on Euclidean distance in 3D PCA space
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

    // Apply PCA to reduce 40D MFCCs to 3D coordinates
    const framesWithPCA = applyPCA(frames);

    const verseSegments = segmentIntoVerses(framesWithPCA, duration);

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
