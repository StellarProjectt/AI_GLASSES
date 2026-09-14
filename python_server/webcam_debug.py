#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HomeSight — Webcam Debug Tool
===============================
Uses your camera (MacBook webcam OR iPhone) to test the full detection
pipeline in real time. No ESP32 needed — runs entirely on your Mac.

iPhone Camera Options
---------------------
  Option A — Continuity Camera (recommended, zero setup):
    • iPhone must be on same WiFi/Bluetooth as Mac
    • macOS 13 Ventura+ and iOS 16+ required
    • iPhone appears automatically as a system camera
    • Run:  python webcam_debug.py --iphone

  Option B — RTSP stream (any iPhone camera app):
    • Install a free app: "Camo" / "EpocCam" / "IP Camera Lite"
    • Copy the RTSP URL shown in the app
    • Run:  python webcam_debug.py --url rtsp://192.168.x.x:8080/live

  Option C — Manual index:
    • Run:  python webcam_debug.py --scan   (see all cameras)
    • Run:  python webcam_debug.py --camera 1

Bounding Box Colours
---------------------
  RED    = P0 Head Hazard   (cabinet, door at eye level)
  AMBER  = P1 Floor Obstacle (chair, fan, box on floor)
  CYAN   = P2 Dynamic Object (person, pet)
  GREEN  = SAFE (detected but suppressed — silent)

Controls
--------
  Q / Esc  — quit
  SPACE    — freeze/unfreeze frame
  S        — speak current alert (macOS say)
  +/-      — raise/lower confidence threshold live

Usage:
  python webcam_debug.py                          # MacBook webcam
  python webcam_debug.py --iphone                 # iPhone via Continuity Camera
  python webcam_debug.py --url rtsp://x.x.x.x/   # RTSP stream
  python webcam_debug.py --scan                   # list all cameras
  python webcam_debug.py --camera 1 --conf 0.30
"""

import argparse
import collections
import subprocess
import sys
import time
import threading
import re
from pathlib import Path
from typing import Optional

# ── OpenCV ───────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("OpenCV not found. Install it:\n  pip install opencv-python")
    sys.exit(1)

# ── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("ultralytics not installed:\n  pip install ultralytics")
    sys.exit(1)

# ── HomeSight Classifier ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from hazard_classifier import classify, HazardPriority

# ─────────────────────────────────────────────
# Camera enumeration (macOS AVFoundation)
# ─────────────────────────────────────────────

def list_cameras() -> list[dict]:
    """
    List all available cameras on macOS using ffmpeg AVFoundation.
    Returns list of {index, name} dicts sorted by index.
    """
    cameras = []
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=8
        )
        # ffmpeg prints device list to stderr
        output = proc.stderr
        in_video = False
        for line in output.splitlines():
            if "AVFoundation video devices" in line:
                in_video = True
            if "AVFoundation audio devices" in line:
                in_video = False
            if in_video:
                # Format: [AVFoundation indev @ ...] [0] FaceTime HD Camera
                m = re.search(r'\[(\d+)\]\s+(.+)', line)
                if m:
                    cameras.append({
                        "index": int(m.group(1)),
                        "name":  m.group(2).strip()
                    })
    except FileNotFoundError:
        # ffmpeg not installed — try brute-force scan indices 0-5
        print("  (ffmpeg not found — scanning camera indices 0–5)")
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({"index": i, "name": f"Camera {i}"})
                cap.release()
    except Exception as e:
        print(f"  Camera scan error: {e}")
    return cameras


def find_iphone_camera() -> Optional[int]:
    """
    Scan all cameras and return the index of the iPhone / Continuity Camera.

    Continuity Camera on macOS shows as "<Your Name> Camera" (not "MacBook").
    Also checks for known virtual camera apps (Camo, EpocCam, etc.)
    """
    cameras = list_cameras()

    # 1. Explicit match — known keywords
    explicit_kw = ["iphone", "continuity", "isight", "camo", "epocam", "droidcam"]
    for cam in cameras:
        if any(kw in cam["name"].lower() for kw in explicit_kw):
            return cam["index"]

    # 2. Continuity Camera heuristic: any non-system camera
    #    (excludes "MacBook", "Built-in", "FaceTime", "Desk View", "Screen")
    skip_kw = ["macbook", "built-in", "facetime", "screen", "desk view", "display"]
    candidates = [
        c for c in cameras
        if not any(sk in c["name"].lower() for sk in skip_kw)
    ]
    if candidates:
        # Prefer the lowest index (main camera, not desk view)
        candidates.sort(key=lambda c: c["index"])
        return candidates[0]["index"]

    return None


def print_cameras():
    """Print all available cameras — used by --scan flag."""
    print()
    print("  Available cameras (macOS AVFoundation):")
    print("  " + "─" * 48)
    cameras = list_cameras()
    if not cameras:
        print("  No cameras found.")
    for cam in cameras:
        tag = ""
        n = cam["name"].lower()
        if any(k in n for k in ["iphone", "continuity", "isight"]):
            tag = "  ← iPhone (explicit)"
        elif any(k in n for k in ["camo", "epocam", "droidcam"]):
            tag = "  ← iPhone virtual camera app"
        elif "macbook" in n or "facetime" in n or "built-in" in n:
            tag = "  ← MacBook built-in webcam"
        elif "screen" in n or "display" in n or "capture" in n:
            tag = "  ← Screen capture"
        elif "desk view" in n:
            tag = "  ← iPhone Desk View (Continuity)"
        else:
            tag = "  ← iPhone / Continuity Camera  ✓ (use --iphone)"
        print(f"  [{cam['index']}]  {cam['name']}{tag}")
    print()
    print("  Usage:  python webcam_debug.py --camera <index>")
    print("          python webcam_debug.py --iphone    (auto-select iPhone)")
    print()


def open_camera(source) -> cv2.VideoCapture:
    """
    Open a camera by index (int) or RTSP/HTTP URL (str).
    Prints clear error messages and exits on failure.
    """
    if isinstance(source, str):
        # RTSP / HTTP stream
        print(f"  Connecting to stream: {source}")
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # low-latency for live streams
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print()
        print(f"  ❌  Cannot open camera source: {source}")
        if isinstance(source, str):
            print()
            print("  RTSP tips:")
            print("    • Check the app is running on iPhone and streaming")
            print("    • Ensure iPhone and Mac are on the same WiFi network")
            print("    • Try the exact URL shown in the app (copy/paste)")
            print()
            print("  Recommended free apps for iPhone RTSP:")
            print("    • Camo  (camo.app) — best quality, USB or WiFi")
            print("    • EpocCam (Kinoni) — WiFi")
            print("    • IP Camera Lite — shows RTSP URL in app")
        else:
            print()
            print("  Tips:")
            print("    • Run  python webcam_debug.py --scan  to see all cameras")
            print("    • For iPhone Continuity Camera: ensure Bluetooth + WiFi on,")
            print("      iPhone unlocked, near Mac — then try --iphone or --camera 1")
        sys.exit(1)
    return cap


# ─────────────────────────────────────────────
# Colour palette (BGR for OpenCV)
# ─────────────────────────────────────────────
COLOUR = {
    HazardPriority.P0:     (0,   50,  255),  # Red
    HazardPriority.P1:     (0,   165, 255),  # Amber
    HazardPriority.P2:     (255, 200, 0),    # Cyan-yellow
    HazardPriority.NORMAL: (50,  200, 80),   # Green (safe)
}
COLOUR_BG = {
    HazardPriority.P0:     (0,   0,   80),
    HazardPriority.P1:     (0,   40,  80),
    HazardPriority.P2:     (80,  60,  0),
    HazardPriority.NORMAL: (0,   50,  20),
}
PRIORITY_LABEL = {
    HazardPriority.P0:     "P0  HEAD",
    HazardPriority.P1:     "P1  FLOOR",
    HazardPriority.P2:     "P2  DYNAMIC",
    HazardPriority.NORMAL: "SAFE",
}

# ─────────────────────────────────────────────
# Text helpers (OpenCV drawing)
# ─────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO  = cv2.FONT_HERSHEY_PLAIN

def draw_text(
    img, text, x, y,
    scale=0.55, colour=(255,255,255), thickness=1,
    bg=None, pad=4
):
    """Draw text with optional filled background."""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)
    return tw, th

def draw_rect_rounded(img, x1, y1, x2, y2, colour, thickness=2, r=8):
    """Draw a rectangle (OpenCV doesn't have built-in rounded corners — use regular)."""
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

def draw_fill_alpha(img, x1, y1, x2, y2, colour, alpha=0.25):
    """Draw a semi-transparent filled rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ─────────────────────────────────────────────
# Speech (macOS say — non-blocking)
# ─────────────────────────────────────────────
_speak_lock = threading.Lock()
_speak_proc: Optional[subprocess.Popen] = None

def speak(text: str):
    """Fire-and-forget macOS TTS."""
    global _speak_proc
    with _speak_lock:
        if _speak_proc and _speak_proc.poll() is None:
            _speak_proc.terminate()
        _speak_proc = subprocess.Popen(
            ["say", "-v", "Samantha", "-r", "210", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

# ─────────────────────────────────────────────
# Cooldown tracker (mirrors ESP32 logic)
# ─────────────────────────────────────────────
COOLDOWN_S = {
    HazardPriority.P0: 4.0,
    HazardPriority.P1: 6.0,
    HazardPriority.P2: 5.0,
}

class CooldownTracker:
    def __init__(self):
        self._last: dict[HazardPriority, float] = {}

    def is_ready(self, priority: HazardPriority) -> bool:
        if priority == HazardPriority.NORMAL:
            return False
        last = self._last.get(priority, 0)
        return (time.time() - last) >= COOLDOWN_S.get(priority, 5.0)

    def fire(self, priority: HazardPriority):
        self._last[priority] = time.time()

    def remaining(self, priority: HazardPriority) -> float:
        last = self._last.get(priority, 0)
        cd   = COOLDOWN_S.get(priority, 5.0)
        return max(0.0, cd - (time.time() - last))

# ─────────────────────────────────────────────
# Alert history (ring buffer for display)
# ─────────────────────────────────────────────
AlertEntry = collections.namedtuple("AlertEntry", ["ts", "priority", "prompt"])

# ─────────────────────────────────────────────
# Main debug loop
# ─────────────────────────────────────────────

def run(args):
    print()
    print("=" * 60)
    print("  HomeSight — Debug Tool")
    print("  Bounding box colours:")
    print("    RED   = P0 Head Hazard")
    print("    AMBER = P1 Floor Obstacle")
    print("    CYAN  = P2 Dynamic Object")
    print("    GREEN = SAFE (suppressed)")
    print()
    print("  Controls:  Q = quit  |  SPACE = freeze  |  S = speak alert")
    print("             + / -  = raise / lower confidence live")
    print("=" * 60)
    print()

    # ── Resolve camera source ──────────────────────────────────────────────────
    if args.url:
        camera_source = args.url
        camera_label  = f"RTSP  {args.url}"
    elif args.iphone:
        print("  🔍 Scanning for iPhone / Continuity Camera...")
        idx = find_iphone_camera()
        if idx is None:
            print()
            print("  ❌ iPhone camera not found automatically.")
            print()
            print("  Setup checklist:")
            print("    1. iPhone and Mac on the same WiFi + Bluetooth enabled")
            print("    2. macOS 13 Ventura+ and iOS 16+ required")
            print("    3. iPhone unlocked and screen facing you")
            print("    4. In iPhone Settings → General → AirPlay & Handoff")
            print("       → 'Continuity Camera' must be ON")
            print()
            print("  Alternative: Use an app for WiFi streaming:")
            print("    • Camo  (camo.app) — install on Mac + iPhone, best quality")
            print("    • EpocCam — install on Mac + iPhone")
            print("    Then run:  python webcam_debug.py --url <RTSP_URL>")
            print()
            print("  Run  python webcam_debug.py --scan  to list all cameras.")
            sys.exit(1)
        print(f"  ✅ iPhone found at camera index {idx}")
        camera_source = idx
        camera_label  = f"iPhone (index {idx})"
    else:
        camera_source = args.camera
        camera_label  = f"Camera index {args.camera}"

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = args.model
    if not Path(model_path).exists():
        model_path = "models/yolov8n.pt"
    if not Path(model_path).exists():
        print(f"Model not found at '{args.model}'.")
        print("Run: python models/download_model.py")
        sys.exit(1)

    print(f"  Loading YOLO model: {model_path} ...")
    model = YOLO(model_path)
    print(f"  Model  : ready")
    print(f"  Source : {camera_label}")
    print()

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = open_camera(camera_source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not isinstance(camera_source, str):
        cap.set(cv2.CAP_PROP_FPS, 30)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera : {W}×{H}")
    print("  Starting... press Q in the window to quit.\n")

    # ── State ────────────────────────────────────────────────────────────────
    cooldown  = CooldownTracker()
    history   = collections.deque(maxlen=6)   # recent alerts for sidebar
    fps_times = collections.deque(maxlen=20)
    inf_times = collections.deque(maxlen=20)

    current_hazard_prompt    = ""
    current_hazard_priority  = HazardPriority.NORMAL
    alert_count              = 0
    frozen                   = False
    frozen_frame             = None
    speak_enabled            = not args.no_sound
    conf = args.conf   # mutable for live adjustment

    cv2.namedWindow("HomeSight Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HomeSight Debug", W, H)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ─────────────────────────────────────────────────────
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            frozen = not frozen
            if frozen:
                ret, frozen_frame = cap.read()
        elif key == ord("s") and current_hazard_prompt and speak_enabled:
            speak(current_hazard_prompt)
        elif key == ord("+") or key == ord("="):
            conf = min(0.95, conf + 0.05)
            print(f"[CONF] → {conf:.0%}")
        elif key == ord("-"):
            conf = max(0.10, conf - 0.05)
            print(f"[CONF] → {conf:.0%}")

        # ── Frame capture ─────────────────────────────────────────────────────
        if frozen and frozen_frame is not None:
            frame = frozen_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

        frame_h, frame_w = frame.shape[:2]
        t_frame = time.perf_counter()

        # ── YOLO inference ────────────────────────────────────────────────────
        t_inf = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=conf,
            verbose=False,
            imgsz=640,
            stream=False,
        )
        inf_ms = (time.perf_counter() - t_inf) * 1000
        inf_times.append(inf_ms)

        # ── Parse detections ──────────────────────────────────────────────────
        detections = []
        raw_boxes  = []   # (x1,y1,x2,y2,name,det_conf) for drawing
        if results:
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    cls_id   = int(box.cls[0])
                    name     = r.names[cls_id]
                    det_conf = float(box.conf[0])
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                    detections.append({
                        "name": name, "confidence": det_conf,
                        "box": {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
                    })
                    raw_boxes.append((x1,y1,x2,y2,name,det_conf))

        # ── Classify hazards ──────────────────────────────────────────────────
        hazards = classify(detections, (frame_h, frame_w), conf)

        # Map detection name → hazard priority for colouring
        hazard_map: dict[str, HazardPriority] = {}
        for h in hazards:
            hazard_map[h.label] = h.priority

        # ── Drawing: bounding boxes ───────────────────────────────────────────
        for (x1,y1,x2,y2,name,conf) in raw_boxes:
            priority = hazard_map.get(name.lower(), HazardPriority.NORMAL)
            col      = COLOUR[priority]
            col_bg   = COLOUR_BG[priority]
            thick    = 3 if priority >= HazardPriority.P1 else 1

            # Fill with transparent overlay for hazards
            if priority >= HazardPriority.P1:
                draw_fill_alpha(frame, x1, y1, x2, y2, col, alpha=0.12)

            # Border
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, thick)

            # Corner markers for P0 (extra visual weight)
            if priority == HazardPriority.P0:
                L = 20
                for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(frame, (cx,cy), (cx+dx*L,cy), col, 4)
                    cv2.line(frame, (cx,cy), (cx,cy+dy*L), col, 4)

            # Label chip
            pri_txt = PRIORITY_LABEL[priority]
            label   = f"{pri_txt}  {name}  {det_conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            lx, ly = x1, max(y1 - 6, th + 6)
            cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 8, ly + 2), col_bg, -1)
            cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 8, ly + 2), col, 1)
            cv2.putText(frame, label, (lx + 4, ly - 2), FONT, 0.5, col, 1, cv2.LINE_AA)

        # ── Top hazard alert ──────────────────────────────────────────────────
        active_alert = False
        if hazards:
            top = hazards[0]
            if cooldown.is_ready(top.priority):
                cooldown.fire(top.priority)
                current_hazard_prompt   = top.prompt
                current_hazard_priority = top.priority
                alert_count += 1
                history.appendleft(AlertEntry(time.time(), top.priority, top.prompt))
                if speak_enabled:
                    speak(top.prompt)
                print(f"[{PRIORITY_LABEL[top.priority]}] {top.prompt}  "
                      f"({top.est_dist_m:.1f}m  {top.side}  conf={top.conf:.0%})")
            active_alert = True

        # Fade prompt after 5 seconds of no alert
        if not active_alert:
            # Don't clear immediately — keep showing for 5 s
            pass

        # ── Bottom warning banner ─────────────────────────────────────────────
        if current_hazard_prompt:
            pri = current_hazard_priority
            col = COLOUR[pri]
            # Animated edge glow for P0
            if pri == HazardPriority.P0:
                pulse = int(abs((time.time() % 0.8) / 0.4 - 1.0) * 80) + 20
                cv2.rectangle(frame, (0,0), (frame_w-1,frame_h-1),
                              (0, pulse//4, pulse), 3)

            # Banner background
            bh = 58
            draw_fill_alpha(frame, 0, frame_h-bh, frame_w, frame_h, col, alpha=0.30)
            cv2.rectangle(frame, (0, frame_h-bh), (frame_w, frame_h), col, 2)

            # Priority tag
            pri_lbl = PRIORITY_LABEL[pri]
            draw_text(frame, pri_lbl,
                      12, frame_h - bh + 20,
                      scale=0.55, colour=col, thickness=2,
                      bg=(10,10,20), pad=4)

            # Main prompt — large
            draw_text(frame, current_hazard_prompt,
                      12, frame_h - 10,
                      scale=1.1, colour=(255,255,255), thickness=2,
                      bg=(10,10,20), pad=4)

            # Cooldown countdown bar
            rem  = cooldown.remaining(pri)
            total_cd = COOLDOWN_S.get(pri, 5.0)
            frac = 1.0 - rem / total_cd
            bar_w = int((frame_w - 24) * frac)
            cv2.rectangle(frame, (12, frame_h-bh+28), (12+bar_w, frame_h-bh+34), col, -1)
            cv2.rectangle(frame, (12, frame_h-bh+28), (frame_w-12, frame_h-bh+34), col, 1)

        else:
            # CLEAR indicator
            bh = 32
            draw_fill_alpha(frame, 0, frame_h-bh, frame_w, frame_h,
                            (50,200,80), alpha=0.15)
            draw_text(frame, "✓  PATH CLEAR — SILENT",
                      12, frame_h - 8,
                      scale=0.65, colour=(50,200,80), thickness=2,
                      bg=(0,20,0), pad=4)

        # ── HUD — top-right info panel ────────────────────────────────────────
        fps = 1.0 / max((time.perf_counter() - t_frame), 0.001)
        fps_times.append(fps)
        avg_fps  = sum(fps_times) / len(fps_times)
        avg_inf  = sum(inf_times) / len(inf_times)

        hud_lines = [
            f"FPS      {avg_fps:5.1f}",
            f"YOLO     {avg_inf:5.0f} ms",
            f"Dets     {len(raw_boxes):5d}",
            f"Hazards  {len(hazards):5d}",
            f"Alerts   {alert_count:5d}",
            f"Conf     {conf:.0%}",
            f"{'FROZEN' if frozen else 'LIVE':>9}",
        ]
        hud_x = frame_w - 170
        hud_y = 14
        cv2.rectangle(frame, (hud_x - 8, 4), (frame_w - 4, 14 + len(hud_lines)*18 + 4),
                      (10,10,25), -1)
        cv2.rectangle(frame, (hud_x - 8, 4), (frame_w - 4, 14 + len(hud_lines)*18 + 4),
                      (60,60,100), 1)
        for i, line in enumerate(hud_lines):
            col_h = (200,200,255) if i < 6 else (150,220,150)
            cv2.putText(frame, line, (hud_x, hud_y + i*18),
                        FONT_MONO, 1.0, col_h, 1, cv2.LINE_AA)

        # ── Alert history sidebar ─────────────────────────────────────────────
        if history:
            sx    = 8
            sy    = 12
            sw    = 220
            sh    = len(history) * 26 + 10
            draw_fill_alpha(frame, sx-4, sy-4, sx+sw, sy+sh, (10,10,30), alpha=0.65)
            cv2.rectangle(frame, (sx-4, sy-4), (sx+sw, sy+sh), (60,60,100), 1)
            for i, entry in enumerate(history):
                col_h = COLOUR[entry.priority]
                age   = time.time() - entry.ts
                alpha_factor = max(0.3, 1.0 - age / 12.0)
                ey = sy + 8 + i * 26
                tag = PRIORITY_LABEL[entry.priority][:2]
                faded = tuple(int(c * alpha_factor) for c in col_h)
                cv2.putText(frame, f"[{tag}] {entry.prompt[:28]}",
                            (sx + 4, ey + 14),
                            FONT, 0.42, faded, 1, cv2.LINE_AA)

        # ── Priority scale bar (vertical left edge) ───────────────────────────
        scale_h = frame_h - 80
        scale_x = frame_w - 22
        scale_y = 40
        cv2.rectangle(frame, (scale_x, scale_y), (scale_x+14, scale_y+scale_h),
                      (30,30,50), -1)
        cv2.rectangle(frame, (scale_x, scale_y), (scale_x+14, scale_y+scale_h),
                      (60,60,100), 1)

        # Fill based on current highest priority
        if hazards:
            top_pri = hazards[0].priority
            fill_frac = top_pri.value / 3.0
            fill_h    = int(scale_h * fill_frac)
            fill_col  = COLOUR[top_pri]
            cv2.rectangle(frame,
                          (scale_x, scale_y + scale_h - fill_h),
                          (scale_x+14, scale_y + scale_h),
                          fill_col, -1)

        # Scale labels
        for lbl, frac, col_l in [
            ("P0", 1.0, COLOUR[HazardPriority.P0]),
            ("P1", 0.66, COLOUR[HazardPriority.P1]),
            ("P2", 0.33, COLOUR[HazardPriority.P2]),
        ]:
            ly = scale_y + int(scale_h * (1.0 - frac))
            cv2.line(frame, (scale_x-4, ly), (scale_x, ly), col_l, 1)
            cv2.putText(frame, lbl, (scale_x-22, ly+4), FONT, 0.35, col_l, 1)

        # ── Title strip ───────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (frame_w, 28), (8, 8, 20), -1)
        cv2.putText(frame, "HomeSight  Debug  —  Webcam  Test",
                    (10, 19), FONT, 0.58, (160,140,255), 1, cv2.LINE_AA)
        if frozen:
            cv2.putText(frame, "[ FROZEN — SPACE to resume ]",
                        (frame_w//2 - 110, 19), FONT, 0.52, (0,220,255), 1)

        cv2.imshow("HomeSight Debug", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("\nDebug session ended.")
    print(f"  Total alerts fired : {alert_count}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="HomeSight Debug — live camera detection with priority overlay"
    )
    # ── Camera source (mutually exclusive) ───────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument("--iphone",  action="store_true",
                     help="Auto-detect iPhone via Continuity Camera")
    src.add_argument("--url",     type=str,   default=None,
                     help="RTSP/HTTP stream URL  (e.g. rtsp://192.168.1.5:8080/live)")
    src.add_argument("--camera",  type=int,   default=0,
                     help="Camera index (default 0 = built-in MacBook webcam)")

    # ── Utility ──────────────────────────────────────────────────────────────
    p.add_argument("--scan",     action="store_true",
                   help="List all available cameras and exit")

    # ── Display / inference ───────────────────────────────────────────────────
    p.add_argument("--width",    type=int,   default=1280)
    p.add_argument("--height",   type=int,   default=720)
    p.add_argument("--conf",     type=float, default=0.40,
                   help="YOLO confidence threshold (default 0.40)")
    p.add_argument("--model",    type=str,   default="models/yolov8n.pt",
                   help="Path to YOLO model (.pt or .onnx)")
    p.add_argument("--no-sound", action="store_true",
                   help="Disable macOS 'say' audio alerts")

    args = p.parse_args()

    if args.scan:
        print_cameras()
        sys.exit(0)

    run(args)


if __name__ == "__main__":
    main()
