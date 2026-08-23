"""
AI Glasses — Video Simulator
==============================
จำลองการทำงานของแว่น AI สำหรับผู้พิการทางสายตา
โดยใช้วิดีโอ POV การเดินแทนกล้องจริง

ระบบย่อย:
  1. YOLO11 — ตรวจจับสิ่งกีดขวาง Real-time (ทุกเฟรม)
  2. Gemini — บรรยายภาพ (กดปุ่ม D)
  3. Gemini — อ่านข้อความ/ป้าย OCR (กดปุ่ม R)
  4. Gemini — ค้นหาสิ่งของ (กดปุ่ม F)
  5. ควบคุมวิดีโอ (Space, ←→, +-, Q)

Usage:
  pip install -r requirements.txt
  python video_simulator.py
"""

import os
import sys

# Add local libs path (packages installed on E: drive due to C: being full)
_libs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if os.path.exists(_libs_path) and _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

import cv2
import numpy as np
import time
import threading
import io
import base64
import math
import json
import tkinter as tk
from pathlib import Path

# Windows beep (built-in, no install needed)
try:
    import winsound
    beep_available = True
except ImportError:
    beep_available = False

# ==========================================
# Configuration
# ==========================================
VIDEO_PATH = str(Path(__file__).parent.parent / "walking_video.mp4")
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "***REMOVED-LEAKED-GEMINI-KEY***"
)

# YOLO settings
YOLO_MODEL = "yolo11n.pt"       # nano model (~6MB, fastest)
YOLO_CONFIDENCE = 0.40          # minimum confidence threshold
YOLO_IOU = 0.45                 # NMS IoU threshold
YOLO_PROCESS_EVERY = 1          # run YOLO every N frames (1=every frame, 3=faster)
YOLO_INPUT_SIZE = 640           # downscale input for YOLO (lower=faster, 640 is default)

# Danger zone settings
DANGER_SIZE_RATIO = 0.30        # object > 30% of frame = very close (imminent)
WARNING_SIZE_RATIO = 0.15       # object > 15% of frame = getting close
CENTER_ZONE_RATIO = 0.40        # middle 40% of frame = "in your path"

# Head-level hazard (objects white cane can't detect)
HEAD_LEVEL_THRESHOLD = 0.55     # if object's bottom edge (y2) is above 55% of frame height
                                # it's likely at head/chest level = dangerous!

# Vehicle classes (need approach detection — don't warn if stationary)
VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
}

# Always-dangerous classes (warn regardless of motion)
ALWAYS_DANGEROUS = {
    "dog", "horse", "cow", "elephant",  # animals can move unpredictably
}

# Approach detection settings
APPROACH_GROWTH_RATE = 0.03     # bbox area grew >3% per frame = approaching
APPROACH_HISTORY = 8            # track over N frames
MIN_WARN_SIZE = 0.05            # ignore objects smaller than 5% of frame

# TTC (Time-To-Collision) settings
TTC_CRITICAL_SEC = 2.0          # TTC < 2s = immediate danger
TTC_WARNING_SEC = 5.0           # TTC < 5s = warning

# Alert cooldown (prevent spam)
ALERT_COOLDOWN_SEC = 4.0
BEEP_COOLDOWN_SEC = 0.5         # beep can repeat faster than TTS

# Heartbeat (path-clear indicator)
HEARTBEAT_INTERVAL = 3.0        # play heartbeat every 3s when path is clear

# Gemini rate limiting
GEMINI_COOLDOWN_SEC = 2.0

# ==========================================
# Colors (BGR for OpenCV)
# ==========================================
COLOR_SAFE = (0, 200, 0)        # Green
COLOR_WARNING = (0, 180, 255)   # Orange
COLOR_DANGER = (0, 0, 255)      # Red
COLOR_INFO = (255, 200, 0)      # Cyan/Light blue
COLOR_TEXT_BG = (30, 30, 30)    # Dark gray
COLOR_WHITE = (255, 255, 255)
COLOR_OVERLAY = (0, 0, 0)       # Black for overlay

# Thai labels for COCO classes
THAI_LABELS = {
    "person": "คน", "bicycle": "จักรยาน", "car": "รถยนต์",
    "motorcycle": "มอเตอร์ไซค์", "airplane": "เครื่องบิน", "bus": "รถบัส",
    "train": "รถไฟ", "truck": "รถบรรทุก", "boat": "เรือ",
    "traffic light": "ไฟจราจร", "fire hydrant": "หัวดับเพลิง",
    "stop sign": "ป้ายหยุด", "parking meter": "มิเตอร์จอดรถ",
    "bench": "ม้านั่ง", "bird": "นก", "cat": "แมว", "dog": "หมา",
    "horse": "ม้า", "sheep": "แกะ", "cow": "วัว", "elephant": "ช้าง",
    "bear": "หมี", "zebra": "ม้าลาย", "giraffe": "ยีราฟ",
    "backpack": "กระเป๋าเป้", "umbrella": "ร่ม", "handbag": "กระเป๋าถือ",
    "tie": "เนคไท", "suitcase": "กระเป๋าเดินทาง", "frisbee": "จานร่อน",
    "skis": "สกี", "snowboard": "สโนว์บอร์ด", "sports ball": "ลูกบอล",
    "kite": "ว่าว", "baseball bat": "ไม้เบสบอล",
    "baseball glove": "ถุงมือเบสบอล", "skateboard": "สเก็ตบอร์ด",
    "surfboard": "กระดานโต้คลื่น", "tennis racket": "ไม้เทนนิส",
    "bottle": "ขวด", "wine glass": "แก้วไวน์", "cup": "แก้วน้ำ",
    "fork": "ส้อม", "knife": "มีด", "spoon": "ช้อน", "bowl": "ชาม",
    "banana": "กล้วย", "apple": "แอปเปิ้ล", "sandwich": "แซนวิช",
    "orange": "ส้ม", "broccoli": "บรอกโคลี", "carrot": "แครอท",
    "hot dog": "ฮอทดอก", "pizza": "พิซซ่า", "donut": "โดนัท",
    "cake": "เค้ก", "chair": "เก้าอี้", "couch": "โซฟา",
    "potted plant": "กระถางต้นไม้", "bed": "เตียง",
    "dining table": "โต๊ะอาหาร", "toilet": "ห้องน้ำ", "tv": "ทีวี",
    "laptop": "แล็ปท็อป", "mouse": "เมาส์", "remote": "รีโมท",
    "keyboard": "คีย์บอร์ด", "cell phone": "โทรศัพท์", "microwave": "ไมโครเวฟ",
    "oven": "เตาอบ", "toaster": "เครื่องปิ้งขนมปัง", "sink": "อ่างล้างจาน",
    "refrigerator": "ตู้เย็น", "book": "หนังสือ", "clock": "นาฬิกา",
    "vase": "แจกัน", "scissors": "กรรไกร", "teddy bear": "ตุ๊กตาหมี",
    "hair drier": "เครื่องเป่าผม", "toothbrush": "แปรงสีฟัน",
    "hole": "หลุม หรือทางยกระดับ",
}


# ==========================================
# Initialize YOLO
# ==========================================
print("=" * 55)
print("  🕶️  AI Glasses — Video Simulator")
print("=" * 55)
print()

print("🔄 กำลังโหลดโมเดล YOLO11...")
from ultralytics import YOLO
yolo_model = YOLO(YOLO_MODEL)
print("✅ YOLO11 พร้อมใช้งาน")

# ==========================================
# Initialize Gemini
# ==========================================
gemini_model = None
try:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-3.6-flash")
    print("✅ Gemini Flash พร้อมใช้งาน")
except Exception as e:
    print(f"⚠️  Gemini ไม่พร้อม: {e}")
    print("   ระบบบรรยายภาพ/OCR จะไม่ทำงาน แต่ YOLO ยังใช้ได้")

# ==========================================
# Initialize TTS (optional)
# ==========================================
tts_available = False
try:
    from gtts import gTTS
    import tempfile
    tts_available = True
    print("✅ TTS (เสียงพูดไทย) พร้อมใช้งาน")
except ImportError:
    print("⚠️  gTTS ไม่พร้อม — จะไม่มีเสียงพูด")


# ==========================================
# Gemini Prompts
# ==========================================
DESCRIBE_PROMPT = """คุณเป็นระบบช่วยเหลือคนตาบอดที่ติดอยู่กับแว่นตา
อธิบายสิ่งที่เห็นในภาพนี้ให้คนตาบอดฟัง

กฎ:
1. ใช้ภาษาไทย กระชับ ชัดเจน
2. บอกตำแหน่ง (ซ้าย กลาง ขวา ข้างหน้า ด้านบน)
3. เรียงลำดับจากสิ่งที่ใกล้ที่สุดและสำคัญที่สุดก่อน
4. ถ้ามีอันตราย ให้เตือนก่อนเสมอ
5. ความยาวไม่เกิน 3 ประโยค
6. ห้ามพูดว่า "ในภาพ" — พูดเหมือนกำลังเดินอยู่จริงๆ"""

OCR_PROMPT = """คุณเป็นระบบช่วยเหลือคนตาบอดที่ติดอยู่กับแว่นตา
อ่านตัวหนังสือ ป้าย ข้อความ ราคา ชื่อร้าน หรือตัวเลข ทุกอย่างที่มีในภาพนี้

กฎ:
1. อ่านข้อความทั้งหมดที่เห็น ทั้งภาษาไทยและภาษาอังกฤษ
2. ถ้ามีหลายข้อความ เรียงจากตัวใหญ่ที่สุด/เด่นที่สุดก่อน
3. บอกตำแหน่งของข้อความ (ซ้าย กลาง ขวา บน ล่าง)
4. ถ้าไม่มีข้อความ ให้บอกว่า "ไม่พบข้อความในภาพ"
5. ใช้ภาษาไทย กระชับ"""

FIND_PROMPT_TEMPLATE = """คุณเป็นระบบช่วยเหลือคนตาบอดที่ติดอยู่กับแว่นตา
คนตาบอดกำลังหา "{target}" อยู่

ดูภาพนี้แล้วตอบ:
1. ถ้าเห็น "{target}" — บอกตำแหน่ง (ซ้าย กลาง ขวา) และระยะโดยประมาณ
   ตัวอย่าง: พบ{target}อยู่ทางขวา ห่างประมาณ 3 เมตร
2. ถ้าไม่เห็น — ตอบ: ยังไม่พบ{target}

ตอบสั้นๆ 1 ประโยค ภาษาไทย"""


# ==========================================
# Helper Functions
# ==========================================
def get_thai_label(eng_label):
    """Get Thai translation of COCO class name"""
    return THAI_LABELS.get(eng_label, eng_label)


def get_zone(cx, frame_width):
    """Determine which zone (left/center/right) an object is in"""
    left_bound = frame_width * (0.5 - CENTER_ZONE_RATIO / 2)
    right_bound = frame_width * (0.5 + CENTER_ZONE_RATIO / 2)
    if cx < left_bound:
        return "ซ้าย", "left"
    elif cx > right_bound:
        return "ขวา", "right"
    else:
        return "กลาง", "center"


def get_clock_position(cx, cy, frame_width, frame_height):
    """Get clock position (1-12) for precise direction guidance for blind users"""
    # Normalize to -1..1 from center
    nx = (cx / frame_width - 0.5) * 2   # -1=far left, +1=far right
    ny = (cy / frame_height - 0.5) * 2  # -1=top, +1=bottom

    import math
    angle = math.degrees(math.atan2(nx, -ny))  # 0=12 o'clock, 90=3 o'clock
    if angle < 0:
        angle += 360
    clock = round(angle / 30) % 12
    if clock == 0:
        clock = 12
    return clock


def estimate_distance(size_ratio):
    """Rough distance estimate based on bounding box size ratio"""
    if size_ratio > 0.40:
        return "ใกล้มาก (ไม่ถึง 1 เมตร)", 0.5
    elif size_ratio > 0.25:
        return "ใกล้ (ประมาณ 1-2 เมตร)", 1.5
    elif size_ratio > 0.12:
        return "ปานกลาง (ประมาณ 2-4 เมตร)", 3.0
    elif size_ratio > 0.05:
        return "ไกลปานกลาง (ประมาณ 4-6 เมตร)", 5.0
    else:
        return "ไกล (มากกว่า 6 เมตร)", 8.0


def get_walking_guidance(clock_pos, distance_m):
    """Generate walking guidance text for blind users"""
    # Direction guidance
    if clock_pos in [11, 12, 1]:
        direction = "ตรงข้างหน้า"
    elif clock_pos in [2, 3]:
        direction = "หันขวาเล็กน้อย"
    elif clock_pos in [4, 5]:
        direction = "หันขวามาก"
    elif clock_pos in [9, 10]:
        direction = "หันซ้ายเล็กน้อย"
    elif clock_pos in [7, 8]:
        direction = "หันซ้ายมาก"
    else:
        direction = "ด้านหลัง"

    # Step estimation (1 step ≈ 0.7m)
    steps = max(1, int(distance_m / 0.7))
    return f"{direction} เดินไปประมาณ {steps} ก้าว"


def get_size_ratio(box_area, frame_area):
    """Calculate how much of the frame the object occupies"""
    return box_area / frame_area if frame_area > 0 else 0


# ==========================================
# Object Approach Tracker (with Ego-Motion Compensation)
# ==========================================
class ObjectTracker:
    """Track objects across frames to detect REAL approaching movement.
    
    Key problem: When the user WALKS FORWARD, the camera moves too,
    making ALL objects in the frame appear to grow larger. A naive
    tracker would think everything is "approaching" even though
    stationary objects (parked cars, benches) are just appearing
    bigger because WE are moving.
    
    Solution: Ego-Motion Compensation
    1. Calculate each object's raw growth rate
    2. Calculate the AVERAGE growth rate of ALL objects (= our walking speed)
    3. Subtract the average (baseline) from each object's rate
    4. Only objects growing FASTER than the baseline are truly approaching
    
    Example:
      - We walk forward → everything grows ~2%/frame (baseline)
      - Parked car grows 2% → relative rate = 0% → SAFE (stationary)
      - Moving car grows 8% → relative rate = 6% → WARNING (approaching!)
    """

    def __init__(self, history_len=8):
        self.history_len = history_len
        # key = (label, zone_en), value = list of size_ratios
        self.tracks = {}
        # Store raw growth rates for ego-motion calculation
        self.raw_rates = {}
        self.baseline_rate = 0.0  # Average growth rate (ego-motion)
        self.frame_count = 0

    def update(self, label, zone_en, size_ratio):
        """Update tracker with new detection. Returns RELATIVE approach_rate
        (after subtracting ego-motion baseline)."""
        key = (label, zone_en)
        if key not in self.tracks:
            self.tracks[key] = []

        self.tracks[key].append(size_ratio)

        # Keep only last N frames
        if len(self.tracks[key]) > self.history_len:
            self.tracks[key] = self.tracks[key][-self.history_len:]

        # Calculate raw growth rate for this object
        raw_rate = self._calc_raw_rate(key)
        self.raw_rates[key] = raw_rate

        # Return RELATIVE rate (compensated for ego-motion)
        return raw_rate - self.baseline_rate

    def update_baseline(self):
        """Recalculate ego-motion baseline from ALL tracked objects.
        Call this AFTER processing all detections in a frame."""
        if not self.raw_rates:
            self.baseline_rate = 0.0
            return

        rates = list(self.raw_rates.values())
        
        if len(rates) >= 2:
            # Use MEDIAN (more robust than mean against outliers)
            sorted_rates = sorted(rates)
            mid = len(sorted_rates) // 2
            if len(sorted_rates) % 2 == 0:
                self.baseline_rate = (sorted_rates[mid - 1] + sorted_rates[mid]) / 2
            else:
                self.baseline_rate = sorted_rates[mid]
        elif len(rates) == 1:
            # Only one object — can't distinguish ego-motion from approach
            # Be conservative: assume it's ego-motion
            self.baseline_rate = rates[0]
        else:
            self.baseline_rate = 0.0

        # Clamp baseline to reasonable values (walking speed)
        # A person walks ~1.4m/s, which produces ~1-4% growth per frame
        self.baseline_rate = max(-0.05, min(0.05, self.baseline_rate))

    def _calc_raw_rate(self, key):
        """Calculate raw growth rate for an object (before ego-motion compensation)"""
        history = self.tracks.get(key, [])
        if len(history) < 3:
            return 0.0  # Not enough data yet

        # Compare average of first half vs second half
        mid = len(history) // 2
        first_half = history[:mid]
        second_half = history[mid:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if first_avg < 0.001:
            return 0.0

        # Growth rate (positive = getting bigger in frame)
        return (second_avg - first_avg) / first_avg

    def get_relative_rate(self, label, zone_en):
        """Get the ego-motion-compensated approach rate for an object."""
        key = (label, zone_en)
        raw = self.raw_rates.get(key, 0.0)
        return raw - self.baseline_rate

    def cleanup(self):
        """Remove stale tracks (objects no longer visible)"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Every ~1 second
            stale = [k for k, v in self.tracks.items() if len(v) > 0 and
                     self.frame_count - len(v) > 60]
            for k in stale:
                del self.tracks[k]
                self.raw_rates.pop(k, None)


def is_head_level(y2, frame_height):
    """Check if object is at head/chest level (white cane can't detect).
    Objects whose bottom edge is in the upper portion of the frame
    are likely hanging/protruding at head height."""
    return (y2 / frame_height) < HEAD_LEVEL_THRESHOLD


def calc_ttc(size_ratio, approach_rate, video_fps=30):
    """Calculate Time-To-Collision in seconds.
    Based on how fast the object is growing in the frame.
    Returns float('inf') if not approaching."""
    if approach_rate <= 0.001:
        return float('inf')  # Not approaching

    # Frames until object fills ~50% of frame (collision threshold)
    target_ratio = 0.50
    if size_ratio >= target_ratio:
        return 0.0  # Already at collision

    # Growth per frame (relative)
    remaining_growth = (target_ratio - size_ratio) / size_ratio
    frames_to_collision = remaining_growth / approach_rate if approach_rate > 0 else float('inf')
    return frames_to_collision / video_fps


def assess_danger(label, zone_en, size_ratio, approach_rate=0.0,
                  y2=0, frame_height=1080, video_fps=30):
    """Assess danger level using size, approach rate, TTC, AND head-level.
    
    Key insights:
    1. Parked car (approach_rate ~ 0) = NOT dangerous unless very close
    2. Head-level objects = ALWAYS dangerous (cane can't detect)
    3. Fast TTC = override cooldown
    """
    # Level: 0=safe, 1=info, 2=warning, 3=danger
    level = 0
    head_level = is_head_level(y2, frame_height)
    ttc = calc_ttc(size_ratio, approach_rate, video_fps)

    # Skip very small objects (far away)
    if size_ratio < MIN_WARN_SIZE:
        # Exception: head-level objects are dangerous even if small
        if head_level and size_ratio > 0.02 and zone_en == "center":
            return 2  # Warn about overhead hazards early
        return 0

    is_approaching = approach_rate > APPROACH_GROWTH_RATE
    is_vehicle = label in VEHICLE_CLASSES
    is_always_dangerous = label in ALWAYS_DANGEROUS

    # === HOLE HAZARD (highest priority for ground) ===
    if label == "hole":
        return 3  # Always max danger for holes

    # === HEAD-LEVEL HAZARD (highest priority!) ===
    # Signs, branches, mirrors, awnings — cane CANNOT detect these
    if head_level and size_ratio > 0.03 and zone_en == "center":
        level = 3  # Always max danger for head-level in path

    # === Animals: always dangerous (unpredictable) ===
    elif is_always_dangerous and size_ratio > MIN_WARN_SIZE:
        level = 2
        if size_ratio > WARNING_SIZE_RATIO:
            level = 3

    # === Vehicles: only dangerous if APPROACHING or VERY close ===
    elif is_vehicle:
        if is_approaching:
            level = 2
            if size_ratio > WARNING_SIZE_RATIO or ttc < TTC_WARNING_SEC:
                level = 3
        elif size_ratio > DANGER_SIZE_RATIO:
            level = 2
        else:
            level = 1

    # === Other objects (person, furniture, etc.) ===
    else:
        if size_ratio > DANGER_SIZE_RATIO:
            level = 3
        elif is_approaching and size_ratio > WARNING_SIZE_RATIO:
            level = 2
        elif size_ratio > WARNING_SIZE_RATIO and zone_en == "center":
            level = 2
        elif size_ratio > MIN_WARN_SIZE:
            level = 1

    # Center zone bump (in your walking path)
    if zone_en == "center" and level == 2:
        level = 3

    # TTC override: imminent collision overrides everything
    if ttc < TTC_CRITICAL_SEC and size_ratio > MIN_WARN_SIZE:
        level = max(level, 3)
    elif ttc < TTC_WARNING_SEC and size_ratio > MIN_WARN_SIZE:
        level = max(level, 2)

    return level


# ==========================================
# Audio Alert System (Beep + TTS)
# ==========================================
def beep_alert(danger_level, zone_en="center"):
    """Play directional beep sound based on danger level.
    Like a car parking sensor: faster beeps = closer danger.
    Uses winsound.Beep (no file needed, instant response)."""
    if not beep_available:
        return

    def _beep():
        try:
            if danger_level >= 3:
                # CRITICAL: rapid high-pitched beeps
                freq = 1200 if zone_en == "center" else (1000 if zone_en == "left" else 1400)
                for _ in range(3):
                    winsound.Beep(freq, 80)   # 80ms beep
                    time.sleep(0.05)           # 50ms gap
            elif danger_level >= 2:
                # WARNING: slower medium beep
                freq = 800 if zone_en == "center" else (700 if zone_en == "left" else 900)
                winsound.Beep(freq, 150)  # Single 150ms beep
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()


def heartbeat_sound():
    """Subtle 'tick' sound to indicate system is alive and path is clear.
    Gives confidence to keep walking."""
    if not beep_available:
        return

    def _tick():
        try:
            winsound.Beep(440, 30)  # Very short, low tone
        except Exception:
            pass

    threading.Thread(target=_tick, daemon=True).start()


# ==========================================
# Dev Debug Overlay
# ==========================================
def draw_dev_debug_box(frame, x1, y1, x2, y2, label, size_ratio,
                       raw_rate, rel_rate, ttc, danger, head_level):
    """Draw detailed debug info below each bounding box for dev inspection."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.35
    color = (200, 200, 200)

    lines = [
        f"Size:{size_ratio:.1%} Raw:{raw_rate:+.1%} Rel:{rel_rate:+.1%}",
        f"Danger:{danger} TTC:{ttc:.1f}s Head:{'Y' if head_level else 'N'}",
    ]

    y_offset = y2 + 14
    for line in lines:
        # Background
        (tw, th), _ = cv2.getTextSize(line, font, fs, 1)
        cv2.rectangle(frame, (x1, y_offset - th - 2), (x1 + tw + 4, y_offset + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, line, (x1 + 2, y_offset), font, fs, color, 1, cv2.LINE_AA)
        y_offset += th + 6


def draw_dev_panel(frame, tracker, fps, last_alert_age, tts_active, beep_active,
                   path_clear_duration):
    """Draw a dev info panel in the top-right corner."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.38
    color = (0, 255, 200)  # Bright cyan-green

    lines = [
        "=== DEV MODE ===",
        f"FPS: {fps:.0f}",
        f"Ego Baseline: {tracker.baseline_rate:+.4f}",
        f"Tracked Objects: {len(tracker.tracks)}",
        f"Alert Cooldown: {last_alert_age:.1f}s ago",
        f"Path Clear: {path_clear_duration:.1f}s",
        f"TTS: {'Active' if tts_active else 'Idle'}",
        f"Beep: {'Active' if beep_active else 'Idle'}",
    ]

    panel_w = 250
    panel_h = len(lines) * 18 + 10
    px = w - panel_w - 10
    py = 60

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    for i, line in enumerate(lines):
        y = py + 16 + i * 18
        c = (0, 255, 255) if i == 0 else color
        cv2.putText(frame, line, (px + 8, y), font, fs, c, 1, cv2.LINE_AA)


def format_time(seconds):
    """Format seconds to MM:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def speak_thai(text):
    """Play Thai TTS in a background thread"""
    if not tts_available:
        return

    def _speak():
        try:
            # Create temp file with unique name
            tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"tts_{int(time.time()*1000)}.mp3")

            tts = gTTS(text=text, lang="th", slow=False)
            tts.save(tmp_path)

            if sys.platform == "win32":
                # Use PowerShell MediaPlayer (doesn't lock the file)
                ps_cmd = (
                    f'powershell -WindowStyle Hidden -Command "'
                    f'Add-Type -AssemblyName PresentationCore; '
                    f'$p = New-Object System.Windows.Media.MediaPlayer; '
                    f'$p.Open([Uri]\'{tmp_path}\'); '
                    f'$p.Play(); '
                    f'Start-Sleep -Seconds 5; '
                    f'$p.Close()"'
                )
                os.system(ps_cmd)

            # Cleanup
            try:
                time.sleep(1)
                os.unlink(tmp_path)
            except:
                pass
        except Exception as e:
            print(f"  TTS Error: {e}")

    threading.Thread(target=_speak, daemon=True).start()


def call_gemini(frame, prompt):
    """Send a frame to Gemini and get response"""
    if gemini_model is None:
        return "Gemini ไม่พร้อมใช้งาน กรุณาตั้ง API Key"

    try:
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Resize to save bandwidth (max 640px wide)
        max_w = 640
        if pil_image.width > max_w:
            ratio = max_w / pil_image.width
            new_h = int(pil_image.height * ratio)
            pil_image = pil_image.resize((max_w, new_h))

        response = gemini_model.generate_content(
            [prompt, pil_image],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


# ==========================================
# UI Drawing Functions
# ==========================================
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_detection_box(frame, x1, y1, x2, y2, label, confidence, danger_level):
    """Draw a styled bounding box with label"""
    # Color based on danger level
    if danger_level >= 3:
        color = COLOR_DANGER
    elif danger_level >= 2:
        color = COLOR_WARNING
    elif danger_level >= 1:
        color = COLOR_INFO
    else:
        color = COLOR_SAFE

    # Draw box
    thickness = 3 if danger_level >= 2 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw corner markers for dangerous objects
    if danger_level >= 2:
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

    # Label text (English to avoid OpenCV '???' issue)
    text = f"{label.capitalize()} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

    # Label background
    lbl_y = max(y1 - 8, th + 8)
    cv2.rectangle(frame, (x1, lbl_y - th - 8), (x1 + tw + 10, lbl_y + 4), color, -1)
    cv2.putText(frame, text, (x1 + 5, lbl_y - 2), font, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_zone_lines(frame):
    """Draw subtle vertical lines showing left/center/right zones"""
    h, w = frame.shape[:2]
    left_x = int(w * (0.5 - CENTER_ZONE_RATIO / 2))
    right_x = int(w * (0.5 + CENTER_ZONE_RATIO / 2))

    # Dashed lines
    dash_len = 15
    for y in range(0, h, dash_len * 2):
        cv2.line(frame, (left_x, y), (left_x, min(y + dash_len, h)), (100, 100, 100), 1)
        cv2.line(frame, (right_x, y), (right_x, min(y + dash_len, h)), (100, 100, 100), 1)


def draw_top_bar(frame, mode_text, fps, current_time, total_time, speed):
    """Draw the top info bar"""
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "AI Glasses Simulator", (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # Mode
    mode_color = COLOR_SAFE if "DETECT" in mode_text else COLOR_INFO
    cv2.putText(frame, mode_text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)

    # FPS
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(frame, fps_text, (w - 100, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # Time
    time_text = f"{format_time(current_time)} / {format_time(total_time)}  {speed:.1f}x"
    cv2.putText(frame, time_text, (w - 220, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_bottom_bar(frame, alert_text="", alert_level=0, gemini_text=""):
    """Draw the bottom info/alert bar"""
    h, w = frame.shape[:2]
    bar_height = 80 if gemini_text else 50

    # Semi-transparent background
    overlay = frame.copy()
    bar_color = COLOR_OVERLAY
    if alert_level >= 3:
        bar_color = (0, 0, 100)  # Dark red
    elif alert_level >= 2:
        bar_color = (0, 60, 100)  # Dark orange
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), bar_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Alert text
    if alert_text:
        alert_color = COLOR_DANGER if alert_level >= 3 else COLOR_WARNING if alert_level >= 2 else COLOR_SAFE
        prefix = "⚠ " if alert_level >= 2 else "ℹ "
        cv2.putText(frame, prefix + alert_text, (15, h - bar_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 1, cv2.LINE_AA)

    # Gemini response text (wrap if needed)
    if gemini_text:
        # Truncate if too long
        max_chars = w // 10
        display_text = gemini_text[:max_chars] + "..." if len(gemini_text) > max_chars else gemini_text
        cv2.putText(frame, display_text, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INFO, 1, cv2.LINE_AA)

    # Controls hint
    controls = "[D] Describe [R] Read [F] Find [T] Jump [V] Dev [Space] Pause [Q] Quit"
    cv2.putText(frame, controls, (15, h - bar_height + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1, cv2.LINE_AA)


def draw_danger_indicator(frame, direction, level):
    """Draw a flashing danger indicator on the side of the frame"""
    h, w = frame.shape[:2]

    if level < 2:
        return

    color = COLOR_DANGER if level >= 3 else COLOR_WARNING
    alpha = 0.4 if level >= 3 else 0.25

    overlay = frame.copy()
    indicator_w = 40

    if direction == "left":
        cv2.rectangle(overlay, (0, 0), (indicator_w, h), color, -1)
    elif direction == "right":
        cv2.rectangle(overlay, (w - indicator_w, 0), (w, h), color, -1)
    elif direction == "center":
        # Top and bottom bars for center danger
        cv2.rectangle(overlay, (0, 0), (w, 15), color, -1)
        cv2.rectangle(overlay, (0, h - 15), (w, h), color, -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def detect_holes_opencv(frame):
    """Simulate hole/puddle detection using OpenCV on the bottom part of the frame"""
    h, w = frame.shape[:2]
    
    # Focus ONLY on the bottom 40% AND the center 50% (the immediate walking path)
    roi_top = int(h * 0.6)
    roi_left = int(w * 0.25)
    roi_right = int(w * 0.75)
    
    roi = frame[roi_top:h, roi_left:roi_right]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # 1. Edge detection for sudden drops/cracks
    edges = cv2.Canny(blurred, 20, 80)
    
    # 2. Adaptive threshold for dark spots (potholes are usually darker)
    dark_spots = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 15)
    
    # Combine edges and dark spots
    combined = cv2.bitwise_or(edges, dark_spots)
    
    # Morphology to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    holes = []
    # Area must be significant relative to ROI size
    roi_area = (h - roi_top) * (roi_right - roi_left)
    min_area = roi_area * 0.05  # At least 5% of the ROI path
    max_area = roi_area * 0.60  # Not the entire screen
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            
            aspect_ratio = w_box / float(h_box)
            
            # Holes usually appear wider than tall due to perspective, 
            # and aren't extremely long/thin like lane lines or walking canes
            if 0.5 < aspect_ratio < 4.0: 
                # Adjust back to global frame coordinates
                global_x = x + roi_left
                global_y = y + roi_top
                holes.append((global_x, global_y, global_x + w_box, global_y + h_box))
                
    return holes

# ==========================================
# Realtime JSON Analysis Engine
# ==========================================
class JSONViewer:
    def __init__(self):
        self.root = None
        self.text_widget = None
        self.ready = False

    def run(self):
        self.root = tk.Tk()
        self.root.title("Spatial Analysis Engine (Realtime JSON)")
        self.root.geometry("600x800")
        
        self.text_widget = tk.Text(self.root, wrap=tk.NONE, font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4")
        self.text_widget.pack(expand=True, fill='both')
        
        v_scroll = tk.Scrollbar(self.text_widget, orient='vertical', command=self.text_widget.yview)
        v_scroll.pack(side='right', fill='y')
        self.text_widget.configure(yscrollcommand=v_scroll.set)
        
        self.ready = True
        self.root.mainloop()

    def update_json(self, json_data):
        if self.ready and self.text_widget:
            try:
                y_pos = self.text_widget.yview()
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(tk.END, json.dumps(json_data, indent=2))
                self.text_widget.yview_moveto(y_pos[0])
            except Exception:
                pass

def generate_analysis_json(frame, all_detections, fps, frame_count, current_max_danger):
    """Generates a deterministic JSON matching the Master Prompt schema based on local data."""
    h, w = frame.shape[:2]
    
    analysis = {
        "system": {
            "analysis_timestamp": time.time(),
            "model_name": "YOLO11 + OpenCV Tracker",
            "model_version": "1.0",
            "processing_mode": "local_realtime",
            "sensor_type": "camera",
            "units": {"distance": "meter", "velocity": "meter_per_second", "time": "second", "angle": "degree"}
        },
        "frame": {
            "frame_id": frame_count,
            "image_width_px": w,
            "image_height_px": h,
            "fps": fps,
            "timestamp": time.time(),
        },
        "scene": {
            "object_count": 0, "dynamic_object_count": 0, "static_object_count": 0,
            "visibility": {"overall_visibility": "good", "lighting_condition": "daylight"},
            "environment": {"indoor_or_outdoor": "unknown", "road_or_path_detected": True},
            "free_space": {"forward_path_clear": True},
            "closest_object_id": None, "most_dangerous_object_id": None
        },
        "objects": [],
        "alerts": [],
        "uncertainty": {
            "overall_confidence": 0.8,
            "detection_confidence": 0.9,
            "depth_confidence": 0.3, 
            "uncertain_factors": ["monocular_depth", "no_lidar"]
        },
        "global_state": {
            "environment_status": "CLEAR" if current_max_danger == 0 else "CAUTION" if current_max_danger == 1 else "HAZARDOUS" if current_max_danger == 2 else "CRITICAL",
            "collision_risk": round(current_max_danger / 3.0, 2),
            "path_status": "CLEAR" if current_max_danger < 2 else "PARTIALLY_BLOCKED",
            "system_action": "CONTINUE" if current_max_danger < 2 else "WARNING",
            "emergency": current_max_danger >= 3
        }
    }
    
    obj_id = 0
    min_dist = 999
    
    for (x1, y1, x2, y2, label, conf, danger, approach_rate) in all_detections:
        obj_id += 1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w_box, h_box = x2 - x1, y2 - y1
        area = w_box * h_box
        size_ratio = area / (w * h)
        _, zone_en = get_zone(cx, w)
        
        _, dist_m_str = estimate_distance(size_ratio)
        try:
            dist_m = float(str(dist_m_str).replace("~", ""))
        except:
            dist_m = 10.0
            
        if dist_m < min_dist:
            min_dist = dist_m
            analysis["scene"]["closest_object_id"] = obj_id
            
        if danger == current_max_danger and danger > 0:
            analysis["scene"]["most_dangerous_object_id"] = obj_id
            
        if danger >= 2 and zone_en == "center":
            analysis["scene"]["free_space"]["forward_path_clear"] = False
                
        is_dynamic = label in VEHICLE_CLASSES or label in ALWAYS_DANGEROUS or label == "person"
        if is_dynamic:
            analysis["scene"]["dynamic_object_count"] += 1
        else:
            analysis["scene"]["static_object_count"] += 1
            
        is_approaching = approach_rate > APPROACH_GROWTH_RATE
        
        obj_data = {
            "object_id": obj_id,
            "tracking_id": f"trk_{label}_{obj_id}",
            "classification": {
                "label": label,
                "category": "vehicle" if label in VEHICLE_CLASSES else "person" if label == "person" else "obstacle" if label == "hole" else "unknown",
                "confidence": round(conf, 2)
            },
            "image_position": {
                "bounding_box": {"x_px": int(x1), "y_px": int(y1), "width_px": int(w_box), "height_px": int(h_box)},
                "normalized_center": {"x": round(cx/w, 2), "y": round(cy/h, 2)},
                "image_region": f"{'top' if cy < h/3 else 'bottom' if cy > 2*h/3 else 'middle'}_{zone_en}"
            },
            "spatial_position": {
                "distance_m": dist_m,
                "direction_from_device": "front" if zone_en == "center" else f"front_{zone_en}"
            },
            "motion": {
                "state": "moving" if is_approaching else "stationary",
                "movement_relative_to_device": "approaching" if is_approaching else "stationary_relative"
            },
            "collision_risk": {
                "score": round(danger / 3.0, 2),
                "level": "SAFE" if danger == 0 else "LOW" if danger == 1 else "HIGH" if danger == 2 else "CRITICAL",
                "urgency": "IMMEDIATE" if danger >= 3 else "HIGH" if danger == 2 else "LOW"
            }
        }
        analysis["objects"].append(obj_data)
        
    analysis["scene"]["object_count"] = obj_id
    return analysis

# ==========================================
# Main Simulator
# ==========================================
def main():
    # Check video file
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ ไม่พบไฟล์วิดีโอ: {VIDEO_PATH}")
        print(f"   กรุณาวางไฟล์ walking_video.mp4 ไว้ที่: {Path(VIDEO_PATH).parent}")
        return

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"\n❌ ไม่สามารถเปิดวิดีโอ: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_seconds = total_frames / video_fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height

    print(f"\n📹 วิดีโอ: {VIDEO_PATH}")
    print(f"   ขนาด: {frame_width}x{frame_height}")
    print(f"   FPS: {video_fps:.1f}")
    print(f"   ความยาว: {format_time(total_seconds)}")
    print(f"   จำนวนเฟรม: {total_frames:,}")
    print()
    print("🎮 ปุ่มควบคุม:")
    print("   [D] — บรรยายภาพ (Describe)")
    print("   [R] — อ่านข้อความ/ป้าย (Read/OCR)")
    print("   [F] — ค้นหาสิ่งของ (Find)")
    print("   [T] — ข้ามเวลา (Time Jump)")
    print("   [Space] — หยุดชั่วคราว / เล่นต่อ")
    print("   [→] / [←] — กรอไปข้างหน้า/ย้อนกลับ 10 วินาที")
    print("   [+] / [-] — เพิ่ม/ลดความเร็ว")
    print("   [Q] — ออก")
    print()
    print("▶️  กำลังเริ่ม...\n")

    # Start JSON Viewer Thread
    json_viewer = JSONViewer()
    threading.Thread(target=json_viewer.run, daemon=True).start()

    # State variables
    paused = False
    playback_speed = 1.0
    mode = "DETECTING"
    alert_text = ""
    alert_speak = ""
    alert_level = 0
    gemini_text = ""
    last_alert_time = 0
    last_gemini_time = 0
    find_target = ""
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0
    frame_skip = 0  # Process YOLO every N frames for speed
    frame_count = 0
    cached_detections = []  # store last YOLO results for redraw on skipped frames
    tracker = ObjectTracker(history_len=APPROACH_HISTORY)

    # Create window
    window_name = "AI Glasses Simulator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Resize window to a reasonable size
    display_w = min(frame_width, 1280)
    display_h = int(display_w * frame_height / frame_width)
    cv2.resizeWindow(window_name, display_w, display_h)

    # Gemini task result (from background thread)
    gemini_result = {"text": "", "done": False}
    gemini_busy = False

    def gemini_task(frame_copy, prompt, result_dict):
        """Run Gemini in background thread"""
        result = call_gemini(frame_copy, prompt)
        result_dict["text"] = result
        result_dict["done"] = True

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            run_yolo = (frame_count % YOLO_PROCESS_EVERY == 0)

            # -----------------------------------------------
            # SYSTEM 1: YOLO Object Detection
            # -----------------------------------------------
            current_max_danger = 0
            current_alert_text = ""
            current_alert_speak = ""
            current_danger_dir = ""

            if run_yolo:
                results = yolo_model(
                    frame, conf=YOLO_CONFIDENCE, iou=YOLO_IOU,
                    imgsz=YOLO_INPUT_SIZE, verbose=False
                )
                cached_detections = []

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Extract box info
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = yolo_model.names[cls_id]

                        # Calculate properties
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        box_area = (x2 - x1) * (y2 - y1)
                        size_ratio = get_size_ratio(box_area, frame_area)
                        zone_th, zone_en = get_zone(cx, frame_width)

                        # Track object across frames for approach detection
                        approach_rate = tracker.update(label, zone_en, size_ratio)

                        # Assess danger (now uses approach detection!)
                        danger = assess_danger(label, zone_en, size_ratio, approach_rate)

                        # Draw bounding box
                        draw_detection_box(frame, x1, y1, x2, y2, label, conf, danger)

                        # Track highest danger
                        if danger > current_max_danger:
                            current_max_danger = danger
                            thai_label = get_thai_label(label)
                            dist_text, dist_m = estimate_distance(size_ratio)

                            if danger >= 3:
                                if approach_rate > APPROACH_GROWTH_RATE and label in VEHICLE_CLASSES:
                                    current_alert_text = f"DANGER! {label.upper()} APPROACHING! ({zone_en})"
                                    current_alert_speak = f"ระวัง! {thai_label} พุ่งมาทางนี้ ({zone_th})"
                                else:
                                    current_alert_text = f"DANGER! {label.upper()} VERY CLOSE ({zone_en})"
                                    current_alert_speak = f"ระวัง! {thai_label} ระยะประชิด ({zone_th})"
                            elif danger >= 2:
                                if approach_rate > APPROACH_GROWTH_RATE and label in VEHICLE_CLASSES:
                                    current_alert_text = f"WARNING: {label.upper()} MOVING ({zone_en})"
                                    current_alert_speak = f"ระวัง: มี{thai_label} เคลื่อนที่ ({zone_th})"
                                else:
                                    current_alert_text = f"WARNING: {label.upper()} ({zone_en}) - {dist_m}m"
                                    if zone_en == "center":
                                        current_alert_speak = f"ระวัง: {thai_label} ขวางทาง ({dist_text})"
                                    else:
                                        current_alert_speak = f"ระวัง: {thai_label} ({zone_th}) ระยะ {dist_text}"
                            current_danger_dir = zone_en

                        # -----------------------------------------------
                        # SYSTEM 4: Find Mode — enhanced guidance
                        # -----------------------------------------------
                        if find_target and mode == "FINDING":
                            if find_target.lower() in label.lower() or \
                               find_target in get_thai_label(label):
                                clock = get_clock_position(cx, cy, frame_width, frame_height)
                                dist_text, dist_m = estimate_distance(size_ratio)
                                guidance = get_walking_guidance(clock, dist_m)
                                thai_label = get_thai_label(label)

                                found_msg_th = (f"พบ{thai_label}! "
                                             f"ตำแหน่ง {clock} นาฬิกา "
                                             f"ระยะ{dist_text} "
                                             f"({guidance})")
                                found_msg_en = f">>> FOUND: {label.upper()} @ {clock} o'clock <<<"
                                gemini_text = found_msg_en
                                speak_thai(found_msg_th)
                                print(f"\n🔎 {found_msg_th}\n")
                                find_target = ""
                                mode = "DETECTING"

                        # Cache detection for redraw on skipped frames
                        cached_detections.append((x1, y1, x2, y2, label, conf, danger, approach_rate))

                # --- OpenCV Hole Detection (Running alongside YOLO) ---
                detected_holes = detect_holes_opencv(frame)
                for hx1, hy1, hx2, hy2 in detected_holes:
                    label = "hole"
                    conf = 0.85 # Mock confidence for UI
                    
                    hcx = (hx1 + hx2) / 2
                    box_area = (hx2 - hx1) * (hy2 - hy1)
                    size_ratio = get_size_ratio(box_area, frame_area)
                    zone_th, zone_en = get_zone(hcx, frame_width)
                    
                    danger = assess_danger(label, zone_en, size_ratio)
                    
                    # Draw box (it will be drawn as DANGER/red)
                    draw_detection_box(frame, hx1, hy1, hx2, hy2, label, conf, danger)
                    
                    if danger > current_max_danger:
                        current_max_danger = danger
                        thai_label = get_thai_label(label)
                        current_alert_text = f"DANGER! HOLE DETECTED ({zone_en})"
                        current_alert_speak = f"ระวัง! {thai_label} ({zone_th})"
                        current_danger_dir = zone_en
                        
                    cached_detections.append((hx1, hy1, hx2, hy2, label, conf, danger, 0.0))

            else:
                # Skipped frame: redraw cached boxes only (no YOLO, much faster)
                for (x1, y1, x2, y2, label, conf, danger, app_rate) in cached_detections:
                    draw_detection_box(frame, x1, y1, x2, y2, label, conf, danger)
                    if danger > current_max_danger:
                        current_max_danger = danger

            # Update ego-motion baseline (MUST be called after all objects processed)
            tracker.update_baseline()

            # Cleanup stale tracks
            tracker.cleanup()
            
            # Generate and Update JSON Realtime
            analysis_json = generate_analysis_json(frame, cached_detections, current_fps, frame_count, current_max_danger)
            json_viewer.update_json(analysis_json)

            # Update alert (with cooldown)
            now = time.time()
            if current_max_danger >= 2 and (now - last_alert_time) > ALERT_COOLDOWN_SEC:
                alert_text = current_alert_text
                alert_speak = current_alert_speak
                alert_level = current_max_danger
                last_alert_time = now

                # Speak alert for high danger
                if current_max_danger >= 3:
                    speak_thai(alert_speak)
            elif (now - last_alert_time) > ALERT_COOLDOWN_SEC * 2:
                # Clear old alert
                alert_text = ""
                alert_speak = ""
                alert_level = 0

            # Draw zone lines
            draw_zone_lines(frame)

            # Draw danger indicators on edges
            if current_danger_dir and current_max_danger >= 2:
                draw_danger_indicator(frame, current_danger_dir, current_max_danger)

        else:
            # Paused — keep showing last frame
            # (frame still exists from last iteration)
            pass

        # -----------------------------------------------
        # Check Gemini background result
        # -----------------------------------------------
        if gemini_result["done"]:
            gemini_text = gemini_result["text"]
            gemini_result["done"] = False
            gemini_busy = False
            print(f"\n📝 Gemini: {gemini_text}\n")
            speak_thai(gemini_text)

        # -----------------------------------------------
        # Calculate FPS
        # -----------------------------------------------
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # -----------------------------------------------
        # Draw UI Overlay
        # -----------------------------------------------
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Mode text
        if mode == "FINDING":
            mode_text = f"MODE: FINDING [{find_target}]"
        elif gemini_busy:
            mode_text = "MODE: AI THINKING..."
        elif paused:
            mode_text = "MODE: PAUSED"
        else:
            mode_text = "MODE: DETECTING"

        draw_top_bar(frame, mode_text, current_fps, current_time, total_seconds, playback_speed)
        draw_bottom_bar(frame, alert_text, alert_level, gemini_text)

        # Show frame
        cv2.imshow(window_name, frame)

        # -----------------------------------------------
        # SYSTEM 5: Keyboard Controls
        # -----------------------------------------------
        wait_time = max(1, int((1000 / video_fps) / playback_speed)) if not paused else 50
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == ord('Q'):
            # Quit
            print("\n👋 ออกจากโปรแกรม")
            break

        elif key == ord(' '):
            # Pause / Resume
            paused = not paused
            status = "⏸️  หยุดชั่วคราว" if paused else "▶️  เล่นต่อ"
            print(status)

        elif key == ord('d') or key == ord('D'):
            # SYSTEM 2: Describe scene
            if not gemini_busy and gemini_model:
                print("\n🧠 กำลังบรรยายภาพ...")
                gemini_busy = True
                gemini_result = {"text": "", "done": False}
                t = threading.Thread(
                    target=gemini_task,
                    args=(frame.copy(), DESCRIBE_PROMPT, gemini_result),
                    daemon=True
                )
                t.start()
                mode = "DESCRIBING"
            elif not gemini_model:
                print("⚠️  Gemini ไม่พร้อมใช้งาน")

        elif key == ord('r') or key == ord('R'):
            # SYSTEM 3: Read / OCR
            if not gemini_busy and gemini_model:
                print("\n📖 กำลังอ่านข้อความ...")
                gemini_busy = True
                gemini_result = {"text": "", "done": False}
                t = threading.Thread(
                    target=gemini_task,
                    args=(frame.copy(), OCR_PROMPT, gemini_result),
                    daemon=True
                )
                t.start()
                mode = "READING"
            elif not gemini_model:
                print("⚠️  Gemini ไม่พร้อมใช้งาน")

        elif key == ord('f') or key == ord('F'):
            # SYSTEM 4: Find mode
            if gemini_model:
                paused = True
                print("\n🔎 พิมพ์ชื่อสิ่งที่ต้องการหา (ภาษาไทยหรืออังกฤษ): ", end="")
                target = input().strip()
                if target:
                    find_target = target
                    mode = "FINDING"
                    print(f"   กำลังค้นหา: {target}")
                    # Also ask Gemini to look in current frame
                    gemini_busy = True
                    gemini_result = {"text": "", "done": False}
                    prompt = FIND_PROMPT_TEMPLATE.format(target=target)
                    t = threading.Thread(
                        target=gemini_task,
                        args=(frame.copy(), prompt, gemini_result),
                        daemon=True
                    )
                    t.start()
                paused = False

        elif key == ord('t') or key == ord('T'):
            # SYSTEM 6: Time Jump
            paused = True
            print("\n⏱️ พิมพ์เวลาที่ต้องการข้ามไป (เช่น 12:30 หรือ 5 สำหรับนาทีที่ 5): ", end="")
            target = input().strip()
            if target:
                try:
                    if ":" in target:
                        m, s = target.split(":")
                        total_sec = int(m) * 60 + int(s)
                    else:
                        # Assume input is minutes if no colon
                        total_sec = float(target) * 60

                    # Cap at video length
                    total_sec = min(total_sec, total_seconds)
                    
                    # Jump
                    cap.set(cv2.CAP_PROP_POS_MSEC, total_sec * 1000)
                    
                    # Clear tracker to prevent ego-motion glitches from the jump
                    tracker.tracks.clear()
                    tracker.raw_rates.clear()
                    
                    print(f"   ⏩ ข้ามไปยังเวลา {format_time(total_sec)}")
                except ValueError:
                    print("   ❌ รูปแบบเวลาไม่ถูกต้อง")
            paused = False

        elif key == 83 or key == ord('.'):
            # Right arrow or '.' — skip forward 10 seconds
            new_pos = cap.get(cv2.CAP_PROP_POS_MSEC) + 10000
            cap.set(cv2.CAP_PROP_POS_MSEC, min(new_pos, total_seconds * 1000))
            print(f"⏩ กรอไปข้างหน้า 10 วินาที → {format_time(new_pos / 1000)}")

        elif key == 81 or key == ord(','):
            # Left arrow or ',' — skip backward 10 seconds
            new_pos = max(0, cap.get(cv2.CAP_PROP_POS_MSEC) - 10000)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_pos)
            print(f"⏪ ย้อนกลับ 10 วินาที → {format_time(new_pos / 1000)}")

        elif key == ord('+') or key == ord('='):
            # Speed up
            playback_speed = min(playback_speed + 0.5, 5.0)
            print(f"⏩ ความเร็ว: {playback_speed:.1f}x")

        elif key == ord('-') or key == ord('_'):
            # Slow down
            playback_speed = max(playback_speed - 0.5, 0.5)
            print(f"⏪ ความเร็ว: {playback_speed:.1f}x")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ ปิดโปรแกรมเรียบร้อย")


if __name__ == "__main__":
    main()
