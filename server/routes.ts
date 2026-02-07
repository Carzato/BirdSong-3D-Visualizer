import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { randomUUID } from "crypto";
import path from "path";
import fs from "fs";
import { analyzeAudioFile } from "./audio-analyzer";

const uploadDir = path.join(process.cwd(), "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, uploadDir);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${randomUUID()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 300 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const allowedMimes = [
      "audio/wav",
      "audio/x-wav",
      "audio/mpeg",
      "audio/mp3",
      "audio/mp4",
      "audio/x-m4a",
      "audio/m4a",
    ];
    const allowedExts = [".wav", ".mp3", ".m4a"];
    const ext = path.extname(file.originalname).toLowerCase();
    
    if (allowedMimes.includes(file.mimetype) || allowedExts.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error("Unsupported file format. Please upload WAV, MP3, or M4A files."));
    }
  },
});

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  app.post("/api/analyze", upload.single("audio"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: "No audio file provided",
        });
      }

      const filePath = req.file.path;
      const filename = req.file.filename;

      console.log(`Analyzing audio file: ${filename}`);

      const analysisResult = await analyzeAudioFile(filePath);

      const visualizationData = {
        audioUrl: `/uploads/${filename}`,
        duration: analysisResult.duration,
        sampleRate: analysisResult.sampleRate,
        verses: analysisResult.verses,
      };

      console.log(`Analysis complete: ${analysisResult.verses.length} verses, ${analysisResult.duration.toFixed(2)}s duration`);

      res.json({
        success: true,
        data: visualizationData,
      });
    } catch (error) {
      console.error("Error analyzing audio:", error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : "Failed to analyze audio",
      });
    }
  });

  app.use("/uploads", (req, res, next) => {
    const cleanPath = req.path.replace(/^\//, "");
    const filePath = path.join(uploadDir, cleanPath);
    if (fs.existsSync(filePath)) {
      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes: Record<string, string> = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
      };
      res.setHeader("Content-Type", mimeTypes[ext] || "application/octet-stream");
      res.sendFile(filePath);
    } else {
      next();
    }
  });

  return httpServer;
}
