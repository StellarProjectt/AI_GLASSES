# 👓 HomeSight — Implementation Walkthrough (v3.0)

## What Changed

Upgraded from cloud-dependent AI Glasses prototype to a fully **offline HomeSight MVP** conforming to PRD v1.0.

## Architecture

```
[ OV3660 Camera ]  ── WiFi LAN (no internet) ──►  [ Mac/PC Python Server ]
                                                        YOLOv8n (local)
                                                        HazardClassifier
                                                        P0 / P1 / P2 / Silent
                                                        pyttsx3 WAV cache
                                                   ◄──────────────────────
[ AppLogic — Priority Arbitration ]
  P0 preempts all → earcon HIGH + speech
  P1 cooldown gate → earcon LOW + speech
  P2 directional  → earcon DIR + speech
  NORMAL          → total silence
       │
[ I2S → MAX98357A → Open-Ear / Bone-Conduction ]
```

## Files Changed

### Python Server
- `server.py` — REWRITTEN: offline YOLO, /analyze + /describe + /search
- `hazard_classifier.py` — NEW: 4-tier matrix + prompt grammar
- `audio_cache.py` — NEW: pre-rendered WAV cache + earcon synthesis
- `models/download_model.py` — NEW: one-time YOLOv8n downloader
- `requirements.txt` — UPDATED: removed cloud SDKs, added ultralytics/pyttsx3
- `test_hazard_classifier.py` — NEW: 30 unit tests (all passed ✅)
- `benchmark_latency.py` — NEW: P99 latency vs PRD KPIs

### ESP32 Firmware
- `AI_GLASSES.ino` — REWRITTEN: double-tap, 1.5s scan, HomeSight dashboard
- `Config.h` — UPDATED: P0/P1/P2 thresholds, cooldowns, new modes
- `AppLogic.h/.cpp` — REWRITTEN: priority arbitration + DoubleTapDetector
- `AudioMod.h/.cpp` — REWRITTEN: P0/P1/P2 earcons, WAV feed, proximity clicks
- `CloudAPI.h/.cpp` — UPDATED: new response fields, /search endpoint

## PRD Compliance

| Requirement | Implementation |
|---|---|
| Silence by default | Empty hazard list → zero audio |
| P0 Head < 1.2m | cy<0.60 + dist<1.2m → HIGH earcon |
| P1 Floor 0.8-1.8m | bottom>0.55 + range → LOW earcon |
| P2 Dynamic < 1.5m | person/pet → DIR earcon |
| Sub-300ms latency | LAN YOLO ~80ms + cached WAV ~5ms |
| Double-tap locator | 400ms window, /search endpoint |
| No cloud dependency | 100% local YOLOv8n + pyttsx3 |

## Setup

```bash
cd python_server
pip install -r requirements.txt
python models/download_model.py
python server.py
```

Set `PYTHON_SERVER_IP` in `Config.h` to your computer's local IP, then upload to ESP32.

## Test Results

30/30 unit tests passed in 0.03s ✅

## Phase 2 Backlog
- Wire VL53L1X ToF sensor for real depth
- MicMod + Whisper-tiny for Object Locator voice queries
- Fine-tune YOLOv8n on home-specific objects
- Validate false alarm rate < 8% with 100-frame neutral test set
- Bone-conduction transducer assembly
