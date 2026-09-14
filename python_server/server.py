# -*- coding: utf-8 -*-
"""
HomeSight — Python Server v3.0 (Offline Edge Inference)
=========================================================
100% offline — zero cloud API dependency.

Inference stack:
  YOLOv8n (ONNX / ultralytics)  →  hazard_classifier  →  audio_cache (pyttsx3 WAV)

Endpoints:
  POST /analyze   — Auto hazard detection (continuous, every 1.5 s from ESP32)
  POST /describe  — Scene description on button press (uses YOLO + structured text)
  POST /search    — Object Locator: find a named item and return relative position
  GET  /health    — Server status + latency stats
  GET  /beep      — Alert beep WAV for ESP32 I2S

Audio Format returned:
  WAV PCM — 16 kHz, 16-bit, mono   (plays directly via ESP32 I2S)
  Encoded as base64 in JSON field "audio_base64"

Latency budget (PRD §5.1):
  Image upload           ≤ 40 ms
  YOLO inference         ≤ 80 ms
  Threat arbitration     ≤ 30 ms
  Audio retrieval (cache)≤ 10 ms  (pre-rendered)
  Total target           ≤ 250 ms  (hard cap 300 ms)

Usage:
  # First run — download model
  python models/download_model.py

  # Start server
  python server.py
  (or) SERVER_PORT=5001 python server.py
"""

import os, io, sys, time, base64, struct, math, logging, traceback
from datetime import datetime
from typing import Optional
from flask import Flask, request, jsonify, send_file

# ── HomeSight modules ──────────────────────────────────────────────────────────
from hazard_classifier import classify, HazardPriority, Hazard
from audio_cache import audio_cache, generate_earcon, SAMPLE_RATE, _wrap_wav

# Windows UTF-8 console fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SERVER_PORT       = int(os.environ.get("SERVER_PORT", "5001"))
DEBUG_MODE        = os.environ.get("DEBUG_MODE", "true").lower() == "true"
CONF_THRESHOLD    = float(os.environ.get("CONF_THRESHOLD", "0.40"))
MODEL_PATH        = os.environ.get("MODEL_PATH", "models/yolov8n.onnx")
MODEL_PATH_PT     = os.environ.get("MODEL_PATH_PT", "models/yolov8n.pt")

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HomeSight")

# ─────────────────────────────────────────────
# YOLO Model Loading
# ─────────────────────────────────────────────
yolo_model  = None
model_type  = None   # "ultralytics" | "onnx"

def _load_model():
    global yolo_model, model_type
    from pathlib import Path

    # Try ultralytics first (easiest API, handles pt + onnx)
    if Path(MODEL_PATH).exists() or Path(MODEL_PATH_PT).exists():
        try:
            from ultralytics import YOLO
            path = MODEL_PATH if Path(MODEL_PATH).exists() else MODEL_PATH_PT
            yolo_model = YOLO(path)
            model_type = "ultralytics"
            logger.info(f"[INIT] YOLO loaded via ultralytics: {path}")
            return
        except ImportError:
            logger.warning("[INIT] ultralytics not installed, trying onnxruntime...")
        except Exception as e:
            logger.warning(f"[INIT] ultralytics load failed: {e}")

    # Fallback: raw onnxruntime
    if Path(MODEL_PATH).exists():
        try:
            import onnxruntime as ort
            yolo_model = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            model_type = "onnx"
            logger.info(f"[INIT] YOLO loaded via onnxruntime: {MODEL_PATH}")
            return
        except ImportError:
            logger.error("[INIT] onnxruntime not installed either. Run: pip install onnxruntime")
        except Exception as e:
            logger.error(f"[INIT] onnxruntime load failed: {e}")

    logger.error("[INIT] No YOLO model available! Run: python models/download_model.py")
    logger.warning("[INIT] Server will start in DEMO mode — no real inference.")


def _run_inference(image_bytes: bytes) -> tuple[list[dict], tuple[int,int]]:
    """
    Run YOLO inference on raw JPEG bytes.
    Returns (detections, (frame_h, frame_w)).

    detections: list of dicts with keys name, confidence, box{x1,y1,x2,y2}
    """
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    frame_shape = (img.height, img.width)

    if yolo_model is None:
        # Demo mode: return empty detections
        return [], frame_shape

    if model_type == "ultralytics":
        results = yolo_model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            verbose=False,
            imgsz=640,
        )
        detections = []
        if results:
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name   = r.names[cls_id]
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    detections.append({
                        "name": name,
                        "confidence": conf,
                        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
        return detections, frame_shape

    # onnxruntime path (more involved — resize + normalise)
    import numpy as np
    inp_size  = 640
    img_resized = img.resize((inp_size, inp_size))
    inp = np.array(img_resized, dtype=np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[np.newaxis]   # NCHW

    input_name  = yolo_model.get_inputs()[0].name
    output_name = yolo_model.get_outputs()[0].name
    raw_out = yolo_model.run([output_name], {input_name: inp})[0]

    # YOLOv8 ONNX output: [1, 84, 8400] — (cx,cy,w,h, cls×80)
    detections = []
    preds = raw_out[0].T  # [8400, 84]
    for pred in preds:
        scores = pred[4:]
        cls_id = int(np.argmax(scores))
        conf   = float(scores[cls_id])
        if conf < CONF_THRESHOLD:
            continue
        cx, cy, w, h = pred[:4]
        # Convert from inp_size coords to original pixel coords
        scale_x = img.width  / inp_size
        scale_y = img.height / inp_size
        x1 = (cx - w / 2) * scale_x
        y1 = (cy - h / 2) * scale_y
        x2 = (cx + w / 2) * scale_x
        y2 = (cy + h / 2) * scale_y
        # COCO names (simplified — load from model metadata in production)
        from ultralytics.data.utils import DATASETS  # may not exist in onnx-only env
        name = str(cls_id)  # fallback
        detections.append({
            "name": name,
            "confidence": conf,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
    return detections, frame_shape


# ─────────────────────────────────────────────
# Scene Description (offline, structured)
# ─────────────────────────────────────────────

_SIDE_WORDS = {"left": "on the left", "center": "ahead", "right": "on the right"}

def _build_description(detections: list[dict], frame_shape: tuple) -> str:
    """
    Build a concise scene description from YOLO detections.
    No LLM required — purely rule-based to maintain offline constraint.
    Max 3 sentences; hazards listed first.
    """
    hazards = classify(detections, frame_shape, CONF_THRESHOLD)
    all_objs = [
        d for d in detections
        if float(d.get("confidence", 0)) >= CONF_THRESHOLD
    ]

    if not all_objs:
        return "Path appears clear."

    parts = []

    # Lead with hazards
    for h in hazards[:2]:
        side_str = _SIDE_WORDS.get(h.side, h.side)
        parts.append(f"{h.label.capitalize()} {side_str}")

    # Then list other notable objects (up to 3 more)
    noted = {h.label for h in hazards}
    for d in all_objs:
        if len(parts) >= 3:
            break
        name = d["name"].lower()
        if name not in noted:
            noted.add(name)
            from hazard_classifier import get_side, estimate_distance
            from hazard_classifier import _FRIENDLY_LABELS
            box  = d["box"]
            fh, fw = frame_shape
            cx = ((box["x1"] + box["x2"]) / 2) / fw
            side = get_side(cx)
            fl   = _FRIENDLY_LABELS.get(name, name)
            parts.append(f"{fl.capitalize()} {_SIDE_WORDS.get(side, side)}")

    return ". ".join(parts) + "."


# ─────────────────────────────────────────────
# Object Locator
# ─────────────────────────────────────────────

def _find_object(query: str, detections: list[dict], frame_shape: tuple) -> Optional[str]:
    """
    Find the closest matching detection to the user's voice query.
    Returns a relative-position string, e.g. "On the table, right side."
    Returns None if not found.
    """
    query_lc = query.lower().strip()
    best = None
    best_conf = 0.0

    for d in detections:
        name = d["name"].lower()
        conf = float(d.get("confidence", 0))
        # Fuzzy match: query word contained in name or vice versa
        if query_lc in name or name in query_lc or any(
            q in name for q in query_lc.split()
        ):
            if conf > best_conf:
                best_conf = conf
                best = d

    if best is None:
        return None

    from hazard_classifier import get_side, estimate_distance
    box  = best["box"]
    fh, fw = frame_shape
    cx = ((box["x1"] + box["x2"]) / 2) / fw
    cy = ((box["y1"] + box["y2"]) / 2) / fh
    w  = (box["x2"] - box["x1"]) / fw
    h  = (box["y2"] - box["y1"]) / fh
    side = get_side(cx)
    dist = estimate_distance((cx, cy, w, h))

    side_str = {"left": "left side", "center": "center", "right": "right side"}[side]
    if dist < 0.8:
        dist_str = "very close"
    elif dist < 1.5:
        dist_str = "about a metre away"
    else:
        dist_str = "further back"

    name_str = best["name"].capitalize()
    return f"{name_str} — {side_str}, {dist_str}."


def _generate_proximity_clicks(distance_norm: float) -> bytes:
    """
    Generate a proximity-click tone WAV where click frequency increases
    as distance_norm approaches 0 (object closer).
    distance_norm: 0.0 = very close, 1.0 = far away.
    """
    # Click rate: 2 Hz (far) → 20 Hz (close)
    click_hz = 2 + (1.0 - max(0, min(1, distance_norm))) * 18
    total_dur_ms = 1500   # 1.5 second proximity guide
    n_total = int(SAMPLE_RATE * total_dur_ms / 1000)

    click_period = int(SAMPLE_RATE / click_hz)
    click_len    = min(int(SAMPLE_RATE * 0.015), click_period // 3)  # 15 ms click

    pcm = bytearray(n_total * 2)
    for start in range(0, n_total - click_len, click_period):
        for i in range(click_len):
            t   = i / SAMPLE_RATE
            env = math.sin(math.pi * i / click_len)
            v   = int(math.sin(2 * math.pi * 880 * t) * 0.6 * 32767 * env)
            v   = max(-32768, min(32767, v))
            offset = (start + i) * 2
            if offset + 1 < len(pcm):
                struct.pack_into("<h", pcm, offset, v)

    return _wrap_wav(bytes(pcm))


# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)
_server_start_time = time.time()
_req_counter       = 0
_latency_history   = []   # rolling list of total latencies (ms)

def _req_id() -> str:
    global _req_counter
    _req_counter += 1
    return f"REQ-{_req_counter:04d}"

def _record_latency(ms: float):
    _latency_history.append(ms)
    if len(_latency_history) > 200:
        _latency_history.pop(0)

def _p99_latency() -> float:
    if not _latency_history:
        return 0.0
    import statistics
    s = sorted(_latency_history)
    idx = int(len(s) * 0.99)
    return s[min(idx, len(s) - 1)]


# ─────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "version":      "3.0-homesight",
        "model":        model_type or "none",
        "model_path":   MODEL_PATH,
        "audio_cache":  audio_cache.is_ready,
        "uptime_s":     int(time.time() - _server_start_time),
        "req_count":    _req_counter,
        "p99_ms":       round(_p99_latency(), 1),
        "wav_spec":     f"{SAMPLE_RATE}Hz/16bit/mono",
    })


# ─────────────────────────────────────────────
# /analyze — Main hazard detection loop
# ─────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Primary HomeSight endpoint — called every 1.5 s by ESP32.
    Returns JSON:
      If hazard detected:
        { alert: true, priority: "P0"|"P1"|"P2",
          prompt: "Watch head — cabinet",
          earcon: "HIGH"|"LOW"|"DIR",
          audio_base64: "<wav pcm b64>",
          elapsed: 0.18 }
      If clear:
        { alert: false, elapsed: 0.05 }
    """
    rid   = _req_id()
    t0    = time.time()

    logger.debug(f"[{rid}] POST /analyze")

    # ── Image extraction ───────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image provided", "req_id": rid}), 400

    t_upload = time.time()
    image_data = request.files["image"].read()
    upload_ms  = (time.time() - t_upload) * 1000
    logger.debug(f"[{rid}] Upload: {len(image_data)//1024} KB in {upload_ms:.0f}ms")

    # ── Inference ──────────────────────────────────────────────────────────────
    t_inf = time.time()
    try:
        detections, frame_shape = _run_inference(image_data)
    except Exception as e:
        logger.error(f"[{rid}] Inference error: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({"alert": False, "error": "inference_failed", "req_id": rid})
    inf_ms = (time.time() - t_inf) * 1000
    logger.debug(f"[{rid}] Inference: {len(detections)} detections in {inf_ms:.0f}ms")

    # ── Hazard Classification ──────────────────────────────────────────────────
    t_cls = time.time()
    hazards = classify(detections, frame_shape, CONF_THRESHOLD)
    cls_ms  = (time.time() - t_cls) * 1000

    elapsed = time.time() - t0

    if not hazards:
        _record_latency(elapsed * 1000)
        logger.info(f"[{rid}] CLEAR — {inf_ms:.0f}ms inf / {cls_ms:.0f}ms cls | total {elapsed*1000:.0f}ms")
        return jsonify({"alert": False, "elapsed": round(elapsed, 3), "req_id": rid})

    # ── Top hazard only (priority arbitration happens on ESP32 too) ───────────
    top: Hazard = hazards[0]
    priority_str = top.priority.name  # "P0" | "P1" | "P2"
    logger.warning(f"[{rid}] HAZARD {priority_str} — '{top.prompt}' "
                   f"(conf={top.conf:.2f} dist≈{top.est_dist_m:.1f}m side={top.side})")

    # ── Audio: earcon + speech (from cache — O(1)) ────────────────────────────
    t_audio = time.time()
    audio_wav = audio_cache.get_combined(top.earcon_code, top.prompt)
    audio_ms  = (time.time() - t_audio) * 1000

    elapsed = time.time() - t0
    _record_latency(elapsed * 1000)

    logger.info(f"[{rid}] {priority_str} '{top.prompt}' | "
                f"upload={upload_ms:.0f} inf={inf_ms:.0f} cls={cls_ms:.0f} "
                f"audio={audio_ms:.0f} | TOTAL={elapsed*1000:.0f}ms")

    result = {
        "alert":       True,
        "priority":    priority_str,
        "prompt":      top.prompt,
        "earcon":      top.earcon_code,
        "side":        top.side,
        "dist_m":      round(top.est_dist_m or 0, 2),
        "elapsed":     round(elapsed, 3),
        "req_id":      rid,
    }

    if audio_wav:
        result["audio_base64"] = base64.b64encode(audio_wav).decode("utf-8")
        result["audio_size"]   = len(audio_wav)
        result["audio_format"] = "wav"

    # Include secondary hazards (so ESP32 can log them)
    if len(hazards) > 1:
        result["additional"] = [
            {"priority": h.priority.name, "prompt": h.prompt}
            for h in hazards[1:3]
        ]

    return jsonify(result)


# ─────────────────────────────────────────────
# /describe — Scene description (button press)
# ─────────────────────────────────────────────

@app.route("/describe", methods=["POST"])
def describe():
    """
    Called when user single-presses the action button.
    Returns structured text description + speech audio.
    """
    rid = _req_id()
    t0  = time.time()
    logger.info(f"[{rid}] POST /describe (button press)")

    if "image" not in request.files:
        return jsonify({"error": "No image", "req_id": rid}), 400

    image_data = request.files["image"].read()

    try:
        detections, frame_shape = _run_inference(image_data)
    except Exception as e:
        logger.error(f"[{rid}] Inference error: {e}")
        return jsonify({"description": "Cannot analyse image — try again.", "req_id": rid})

    description = _build_description(detections, frame_shape)
    logger.info(f"[{rid}] Description: \"{description}\"")

    audio_wav = audio_cache.get_speech(description)
    # If description not cached (it won't be for dynamic combos), synthesise
    if audio_wav is None:
        from audio_cache import _synth_with_pyttsx3
        audio_wav = _synth_with_pyttsx3(description)

    elapsed = time.time() - t0
    result = {
        "description": description,
        "elapsed":     round(elapsed, 3),
        "req_id":      rid,
    }
    if audio_wav:
        result["audio_base64"] = base64.b64encode(audio_wav).decode("utf-8")
        result["audio_format"] = "wav"
        result["audio_size"]   = len(audio_wav)

    logger.info(f"[{rid}] /describe done in {elapsed*1000:.0f}ms")
    return jsonify(result)


# ─────────────────────────────────────────────
# /search — Object Locator (double-tap)
# ─────────────────────────────────────────────

@app.route("/search", methods=["POST"])
def search():
    """
    Called when user double-taps the temple and speaks a query.
    Request: multipart with 'image' (JPEG) + 'query' (text string from STT)
    Returns: relative position string + proximity-click WAV
    """
    rid   = _req_id()
    t0    = time.time()
    query = request.form.get("query", "").strip()
    logger.info(f"[{rid}] POST /search — query: '{query}'")

    if not query:
        return jsonify({"error": "No query provided", "req_id": rid}), 400
    if "image" not in request.files:
        return jsonify({"error": "No image", "req_id": rid}), 400

    image_data = request.files["image"].read()

    try:
        detections, frame_shape = _run_inference(image_data)
    except Exception as e:
        logger.error(f"[{rid}] Inference error: {e}")
        return jsonify({"found": False, "message": "Cannot analyse image.", "req_id": rid})

    location = _find_object(query, detections, frame_shape)

    if location is None:
        msg = f"Cannot find {query} in view."
        audio = audio_cache.get_speech(msg)
        if audio is None:
            from audio_cache import _synth_with_pyttsx3
            audio = _synth_with_pyttsx3(msg)
        result = {"found": False, "message": msg, "req_id": rid}
    else:
        # Estimate normalised distance for click tone
        best_dist = 0.5   # default mid-range
        for d in detections:
            if d["name"].lower() in query.lower() or query.lower() in d["name"].lower():
                box = d["box"]
                fh, fw = frame_shape
                cx = ((box["x1"] + box["x2"]) / 2) / fw
                cy = ((box["y1"] + box["y2"]) / 2) / fh
                w  = (box["x2"] - box["x1"]) / fw
                h  = (box["y2"] - box["y1"]) / fh
                from hazard_classifier import estimate_distance
                dist_m = estimate_distance((cx, cy, w, h))
                best_dist = min(1.0, dist_m / 3.0)
                break

        speech_wav = audio_cache.get_speech(location)
        if speech_wav is None:
            from audio_cache import _synth_with_pyttsx3
            speech_wav = _synth_with_pyttsx3(location)

        click_wav = _generate_proximity_clicks(best_dist)

        # Return speech first, then click (combined)
        audio = speech_wav  # ESP32 plays speech, then loops click_wav
        result = {
            "found":         True,
            "message":       location,
            "distance_norm": round(best_dist, 2),
            "req_id":        rid,
        }
        if click_wav:
            result["click_audio_b64"] = base64.b64encode(click_wav).decode("utf-8")

    elapsed = time.time() - t0
    result["elapsed"] = round(elapsed, 3)

    if audio:
        result["audio_base64"] = base64.b64encode(audio).decode("utf-8")
        result["audio_format"] = "wav"

    logger.info(f"[{rid}] /search '{query}' → {result.get('message')} in {elapsed*1000:.0f}ms")
    return jsonify(result)


# ─────────────────────────────────────────────
# /beep — Earcon download
# ─────────────────────────────────────────────

@app.route("/beep", methods=["GET"])
def beep():
    """Return high-priority earcon WAV for ESP32 startup confirmation."""
    wav = audio_cache.get_earcon("HIGH")
    if not wav:
        wav = generate_earcon("HIGH")
    return send_file(io.BytesIO(wav), mimetype="audio/wav", as_attachment=False)


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "path": request.path}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"[500] {e}\n{traceback.format_exc()}")
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print()
    print("=" * 62)
    print("  HomeSight — Python Server v3.0  (Offline Edge Inference)")
    print("  YOLOv8n + pyttsx3 | Zero cloud dependency")
    print("=" * 62)
    print()

    # Load YOLO model
    _load_model()

    # Warm up audio cache in background thread
    audio_cache.warmup(blocking=False)

    print(f"  Config:")
    print(f"    Model       : {MODEL_PATH}  ({model_type or 'not loaded!'})")
    print(f"    Audio cache : warming up in background...")
    print(f"    YOLO conf   : {CONF_THRESHOLD}")
    print(f"    Debug       : {'ON' if DEBUG_MODE else 'OFF'}")
    print()
    print(f"  Endpoints:")
    print(f"    POST /analyze  — Auto hazard detection (every 1.5s)")
    print(f"    POST /describe — Scene description (button press)")
    print(f"    POST /search   — Object locator (double-tap)")
    print(f"    GET  /health   — Status + latency stats")
    print(f"    GET  /beep     — Earcon WAV")
    print()
    print(f"  Server at:")
    print(f"    http://localhost:{SERVER_PORT}")
    print(f"    http://{local_ip}:{SERVER_PORT}")
    print()
    print(f"  ESP32 Config.h:")
    print(f'    #define PYTHON_SERVER_IP    "{local_ip}"')
    print(f"    #define PYTHON_SERVER_PORT  {SERVER_PORT}")
    print()
    print("  Starting Flask... (Ctrl+C to stop)")
    print()

    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)
