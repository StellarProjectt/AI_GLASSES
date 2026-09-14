# -*- coding: utf-8 -*-
"""
HomeSight — Python Server v3.0 (Offline Edge Inference)
=========================================================
100% offline — zero cloud API dependency.

Inference stack:
  YOLOv8s (ONNX / ultralytics)  →  hazard_classifier  →  audio_cache (pyttsx3 WAV)

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
from typing import Optional, Tuple
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
CONF_THRESHOLD    = float(os.environ.get("CONF_THRESHOLD", "0.35"))
# yolov8s ("small") is meaningfully more accurate than the nano model at a
# real but acceptable CPU-latency cost on a desktop/laptop server — worth it
# for a device whose whole job is not walking a blind user into things.
# Set MODEL_PATH/MODEL_PATH_PT to the yolov8n files (or run
# `MODEL_SIZE=n python models/download_model.py`) to trade accuracy back
# for speed on slower hardware.
MODEL_PATH        = os.environ.get("MODEL_PATH", "models/yolov8s.onnx")
MODEL_PATH_PT     = os.environ.get("MODEL_PATH_PT", "models/yolov8s.pt")
# P0 head-level hazards are the most dangerous class (PRD: risk of head/face
# injury) — accept more false positives there in exchange for fewer missed
# detections. Everything else keeps the normal threshold.
CONF_THRESHOLD_P0  = float(os.environ.get("CONF_THRESHOLD_P0", "0.28"))

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
# COCO-80 class names (standard order used by every YOLOv8 COCO checkpoint).
# Needed only by the raw-onnxruntime fallback path, which has no access to
# the ultralytics model metadata that normally supplies these.
# ─────────────────────────────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


def _nms(boxes: "np.ndarray", scores: "np.ndarray", cls_ids: "np.ndarray",
         iou_threshold: float = 0.45) -> list[int]:
    """
    Minimal numpy class-aware Non-Max Suppression.
    Without this, the raw-onnxruntime path emits many overlapping boxes for
    the same physical object (YOLO predicts one box per anchor point), which
    would otherwise spam multiple hazard alerts for a single obstacle.
    Returns indices of kept boxes.
    """
    import numpy as np
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        same_class = cls_ids[rest] == cls_ids[i]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0)
        suppress = same_class & (iou > iou_threshold)
        order = rest[~suppress]
    return keep


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
        # Run inference at the lowest threshold any downstream consumer
        # needs (P0 head hazards use a lower bar than the general default)
        # — classify() re-filters per-category afterwards, so we never lose
        # a real detection to an overly strict threshold applied too early.
        with _inference_lock:
            results = yolo_model.predict(
                source=img,
                conf=min(CONF_THRESHOLD, CONF_THRESHOLD_P0),
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
    inp_size = 640

    # Letterbox (pad to square, keep aspect ratio) instead of a naive
    # stretch-resize — stretching distorts every object's proportions,
    # which measurably hurts a detector's accuracy (and this classifier's
    # own bbox-area distance estimate assumes undistorted geometry).
    scale = min(inp_size / img.width, inp_size / img.height)
    new_w, new_h = int(round(img.width * scale)), int(round(img.height * scale))
    resized = img.resize((new_w, new_h))
    pad_x, pad_y = (inp_size - new_w) // 2, (inp_size - new_h) // 2
    canvas = PILImage.new("RGB", (inp_size, inp_size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))

    inp = np.array(canvas, dtype=np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[np.newaxis]   # NCHW

    input_name  = yolo_model.get_inputs()[0].name
    output_name = yolo_model.get_outputs()[0].name
    with _inference_lock:
        raw_out = yolo_model.run([output_name], {input_name: inp})[0]

    # YOLOv8 ONNX output: [1, 84, 8400] — (cx,cy,w,h, cls×80)
    preds = raw_out[0].T  # [8400, 84]
    conf_floor = min(CONF_THRESHOLD, CONF_THRESHOLD_P0)
    all_scores = preds[:, 4:]
    cls_ids    = all_scores.argmax(axis=1)
    confs      = all_scores[np.arange(len(preds)), cls_ids]
    mask       = confs >= conf_floor
    if not np.any(mask):
        return [], frame_shape

    boxes_cxcywh = preds[mask, :4]
    cls_ids      = cls_ids[mask]
    confs        = confs[mask]

    # Undo letterbox padding/scaling to get back to original pixel coords
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms(boxes_xyxy, confs, cls_ids)

    detections = []
    for i in keep:
        cid = int(cls_ids[i])
        name = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else str(cid)
        bx1, by1, bx2, by2 = boxes_xyxy[i]
        detections.append({
            "name": name,
            "confidence": float(confs[i]),
            "box": {
                "x1": float(max(0, min(bx1, img.width))),
                "y1": float(max(0, min(by1, img.height))),
                "x2": float(max(0, min(bx2, img.width))),
                "y2": float(max(0, min(by2, img.height))),
            },
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
    hazards = classify(detections, frame_shape, CONF_THRESHOLD,
                        conf_threshold_p0=CONF_THRESHOLD_P0)
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

import re as _re

_STOPWORDS = {
    "a", "an", "the", "my", "find", "where", "is", "are", "locate",
    "search", "for", "please", "can", "you", "me", "i", "lost", "of",
}

def _tokenize(s: str) -> set[str]:
    return set(_re.findall(r"[a-z]+", s.lower()))

# Reverse of hazard_classifier._FRIENDLY_LABELS (coco name -> friendly word),
# plus natural-language aliases people actually say that aren't in that map.
def _build_synonyms() -> dict[str, set[str]]:
    from hazard_classifier import _FRIENDLY_LABELS
    syn: dict[str, set[str]] = {}
    for coco_name, friendly in _FRIENDLY_LABELS.items():
        for tok in _tokenize(friendly):
            syn.setdefault(tok, set()).add(coco_name)
    extra = {
        "phone": {"cell phone"}, "mobile": {"cell phone"},
        "seat": {"chair"}, "sofa": {"couch"},
        "television": {"tv"}, "screen": {"tv", "laptop"},
        "bag": {"backpack", "handbag", "suitcase"}, "purse": {"handbag"},
        "glass": {"wine glass", "cup"}, "computer": {"laptop"},
        "drier": {"hair drier"}, "dryer": {"hair drier"},
        "fridge": {"refrigerator"}, "remote control": {"remote"},
    }
    for tok, names in extra.items():
        syn.setdefault(tok, set()).update(names)
    return syn

_SYNONYMS = _build_synonyms()


def _find_object(
    query: str, detections: list[dict], frame_shape: tuple
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Find the best-matching detection for the user's query.
    Returns (relative-position string, the matched detection dict) so the
    caller can reuse the exact same detection for proximity-click distance
    instead of re-matching (and possibly picking a different object).
    Returns (None, None) if nothing plausible is in view.

    Matching is whole-word, not substring — the previous implementation used
    `q in name` for each query word, so a short word like "a" (contained in
    "chair", "table", "lamp", ...) would match almost anything in the room
    and hand back a confident-sounding but essentially random object.
    """
    query_tokens = _tokenize(query) - _STOPWORDS
    if not query_tokens:
        return None, None

    from hazard_classifier import _FRIENDLY_LABELS

    best_score = 0
    best_conf  = 0.0
    best_det   = None

    for d in detections:
        name = str(d.get("name", "")).lower()
        conf = float(d.get("confidence", 0))
        name_tokens = _tokenize(name) | _tokenize(_FRIENDLY_LABELS.get(name, ""))

        if query_tokens & name_tokens:
            score = 2  # direct match against the class name or its friendly label
        elif any(name in _SYNONYMS.get(tok, ()) for tok in query_tokens):
            score = 1  # matched via a natural-language synonym
        else:
            continue

        if score > best_score or (score == best_score and conf > best_conf):
            best_score = score
            best_conf  = conf
            best_det   = d

    if best_det is None:
        return None, None

    from hazard_classifier import get_side, estimate_distance
    box  = best_det["box"]
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

    fl = _FRIENDLY_LABELS.get(best_det["name"].lower(), best_det["name"])
    name_str = fl.capitalize()
    return f"{name_str} — {side_str}, {dist_str}.", best_det


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
import threading

app = Flask(__name__)
# ESP32 sends small JPEGs (VGA/QVGA at moderate quality — tens of KB). Cap
# uploads generously above that so a corrupt/runaway request can't consume
# unbounded memory/CPU; Flask returns 413 automatically past this.
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB

_server_start_time = time.time()
_req_counter       = 0
_latency_history   = []   # rolling list of total latencies (ms)
_stats_lock        = threading.Lock()

# Most inference runtimes (ultralytics/onnxruntime) aren't guaranteed
# thread-safe for concurrent calls on one model instance, and Flask runs
# threaded — serialize inference so the ESP32's /analyze polling and a
# simultaneous web-dashboard /describe or /search never race each other.
_inference_lock = threading.Lock()

def _req_id() -> str:
    global _req_counter
    with _stats_lock:
        _req_counter += 1
        return f"REQ-{_req_counter:04d}"

def _record_latency(ms: float):
    with _stats_lock:
        _latency_history.append(ms)
        if len(_latency_history) > 200:
            _latency_history.pop(0)

def _p99_latency() -> float:
    with _stats_lock:
        if not _latency_history:
            return 0.0
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
# Temporal confirmation — smooths single-frame misdetections
# ─────────────────────────────────────────────
from collections import deque

_RECENT_FRAMES = 2  # plus the current frame = "seen in >=2 of last 3"
_recent_hazard_keys: "deque" = deque(maxlen=_RECENT_FRAMES)
_confirm_lock = threading.Lock()


def _confirm_hazards(hazards: list) -> list:
    """
    Require a P1/P2 hazard to appear in at least 2 of the last 3 /analyze
    frames (this one plus the 2 before it) before it's allowed through.
    /analyze runs on a live, noisy camera every 1.5s — a single misread
    frame (motion blur, bad angle, a momentary misclassification) would
    otherwise become a spoken alert immediately, which reads as the system
    being inaccurate/trigger-happy even when the underlying detector is
    working as intended.

    P0 head-level hazards are exempt and always fire on the first frame:
    for the one category where missing a hit really matters, a half-second
    of extra caution is worse than the false-alarm cost of firing once on a
    misdetection.
    """
    key = lambda h: (h.priority, h.label, h.side)
    current_keys = {key(h) for h in hazards if h.priority != HazardPriority.P0}

    with _confirm_lock:
        prior_sets = list(_recent_hazard_keys)
        _recent_hazard_keys.append(current_keys)

    confirmed = []
    for h in hazards:
        if h.priority == HazardPriority.P0:
            confirmed.append(h)
            continue
        seen_in_prior = sum(1 for s in prior_sets if key(h) in s)
        if seen_in_prior >= 1:
            confirmed.append(h)
    return confirmed


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

    # A model that failed to load must never look like "path is clear" —
    # that's indistinguishable from a genuinely safe scene to the user, who
    # would otherwise walk around trusting hazard detection that isn't
    # actually running. Say so explicitly instead.
    if yolo_model is None:
        elapsed = time.time() - t0
        logger.error(f"[{rid}] No model loaded — refusing to claim 'clear'.")
        return jsonify({
            "alert": False, "system_ready": False,
            "elapsed": round(elapsed, 3), "req_id": rid,
        })

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
        return jsonify({"alert": False, "system_ready": False,
                         "error": "inference_failed", "req_id": rid})
    inf_ms = (time.time() - t_inf) * 1000
    logger.debug(f"[{rid}] Inference: {len(detections)} detections in {inf_ms:.0f}ms")

    # ── Hazard Classification ──────────────────────────────────────────────────
    t_cls = time.time()
    try:
        hazards = classify(detections, frame_shape, CONF_THRESHOLD,
                            conf_threshold_p0=CONF_THRESHOLD_P0)
    except Exception as e:
        logger.error(f"[{rid}] Classification error: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({"alert": False, "system_ready": False,
                         "error": "classify_failed", "req_id": rid})
    hazards = _confirm_hazards(hazards)
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

    if yolo_model is None:
        return jsonify({"description": "Detection is not running — check the server.",
                         "system_ready": False, "req_id": rid})

    image_data = request.files["image"].read()

    try:
        detections, frame_shape = _run_inference(image_data)
        description = _build_description(detections, frame_shape)
    except Exception as e:
        logger.error(f"[{rid}] /describe failed: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({"description": "Cannot analyse image — try again.", "req_id": rid})

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
    if yolo_model is None:
        return jsonify({"found": False, "message": "Detection is not running.",
                         "system_ready": False, "req_id": rid})

    image_data = request.files["image"].read()

    try:
        detections, frame_shape = _run_inference(image_data)
        location, matched_det = _find_object(query, detections, frame_shape)
    except Exception as e:
        logger.error(f"[{rid}] /search failed: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({"found": False, "message": "Cannot analyse image.", "req_id": rid})

    if location is None:
        msg = f"Cannot find {query} in view."
        audio = audio_cache.get_speech(msg)
        if audio is None:
            from audio_cache import _synth_with_pyttsx3
            audio = _synth_with_pyttsx3(msg)
        result = {"found": False, "message": msg, "req_id": rid}
    else:
        # Distance for the click tone comes from the SAME detection the
        # message describes — previously this re-matched independently with
        # a looser substring check, so the click cadence could describe a
        # different object than the one just named out loud.
        best_dist = 0.5   # default mid-range
        if matched_det is not None:
            box = matched_det["box"]
            fh, fw = frame_shape
            cx = ((box["x1"] + box["x2"]) / 2) / fw
            cy = ((box["y1"] + box["y2"]) / 2) / fh
            w  = (box["x2"] - box["x1"]) / fw
            h  = (box["y2"] - box["y1"]) / fh
            from hazard_classifier import estimate_distance
            dist_m = estimate_distance((cx, cy, w, h))
            best_dist = min(1.0, dist_m / 3.0)

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
    print("  YOLOv8s + pyttsx3 | Zero cloud dependency")
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
