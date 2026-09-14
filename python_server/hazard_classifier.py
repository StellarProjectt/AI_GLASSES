# -*- coding: utf-8 -*-
"""
HomeSight — Hazard Classifier
==============================
Implements the 4-tier Hazard Detection Matrix from the PRD:

  P0  Critical Head Hazard  — ajar doors, open cabinets, low-hanging fixtures
  P1  Low Path Obstacle     — chairs, fans, boxes, shoes, cords, pets on floor
  P2  Dynamic Object        — moving persons, approaching pets
  NORMAL                    — walls, closed furniture, clear path → MUTE

Input:  List of YOLO detections (class_name, bbox, confidence, frame_shape)
Output: List[Hazard] sorted by priority (P0 first)

Audio Prompt Grammar (PRD §4.2):
  P0 → "Watch head — {label}"
  P1 → "Watch steps — {label}" | "Watch {side} — {label}"
  P2 → "Pet ahead" | "Person passing"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────
# Priority Levels
# ─────────────────────────────────────────────

class HazardPriority(IntEnum):
    NORMAL = 0   # Silent — no alert
    P2     = 1   # Dynamic object
    P1     = 2   # Floor obstacle
    P0     = 3   # Head-level critical


# ─────────────────────────────────────────────
# Hazard class
# ─────────────────────────────────────────────

@dataclass
class Hazard:
    priority:   HazardPriority
    label:      str           # raw YOLO class name
    prompt:     str           # audio output string
    side:       str           # "left" | "center" | "right"
    conf:       float         # YOLO confidence score
    bbox_norm:  Tuple[float, float, float, float]  # (cx, cy, w, h) normalised 0-1
    # Estimated distance (meters). None until ToF is available;
    # falls back to bounding-box-area heuristic.
    est_dist_m: Optional[float] = None

    @property
    def earcon_code(self) -> str:
        """Short code sent to ESP32 to select the correct earcon tone."""
        return {
            HazardPriority.P0: "HIGH",
            HazardPriority.P1: "LOW",
            HazardPriority.P2: "DIR",
        }.get(self.priority, "NONE")


# ─────────────────────────────────────────────
# YOLO Class → HomeSight Category Mappings
# (based on COCO-80 class names used by YOLOv8n)
# ─────────────────────────────────────────────

# P0 — Head-level hazards
# These objects can be at face/chest height and cause head collisions
_P0_CLASSES: set[str] = {
    "door",          # hinged door (open / ajar) — custom fine-tune later
    "cabinet",       # upper kitchen cabinet
    "refrigerator",  # tall appliance — door can be open at head level
    "microwave",     # mounted microwave
    "tv",            # wall-mounted display
    "laptop",        # on desk at face height when standing close
    "monitor",       # desktop monitor
    # Generic fallback when custom classes aren't present
    "umbrella",      # approximates low-hanging objects
}

# P1 — Floor-level obstacles (below waist)
_P1_CLASSES: set[str] = {
    "chair",
    "bench",
    "dining table",
    "suitcase",
    "backpack",
    "handbag",
    "sports ball",
    "bottle",
    "cup",
    "bowl",
    "vase",
    "potted plant",
    "book",
    "remote",
    "scissors",
    "hair drier",
    "toothbrush",
    "cell phone",
    "keyboard",
    "mouse",
    "skateboard",
    "surfboard",
    "tennis racket",
    "frisbee",
    # Common household items not in COCO — matched by partial string below
    # "fan", "cord", "cable", "box", "shoe"
}

# P1 partial-name matches (catches "electric fan", "floor fan", etc.)
_P1_PARTIALS: list[str] = [
    "fan", "cord", "cable", "wire", "box", "shoe", "boot",
    "bag", "basket", "bin", "trash", "waste"
]

# P2 — Dynamic objects
_P2_CLASSES: set[str] = {
    "person",
    "cat",
    "dog",
    "bird",
    "horse",
    "sheep",
    "cow",
    "bear",
    "zebra",
    "giraffe",
}

# Objects that are universally SAFE (closed furniture, walls, architecture)
_SAFE_CLASSES: set[str] = {
    "wall",
    "ceiling",
    "floor",
    "couch",   # typically large, predictable, already known
    "bed",
    "toilet",
    "sink",
    "oven",    # built-in, stationary
    "clock",
}


# ─────────────────────────────────────────────
# Distance Estimation
# ─────────────────────────────────────────────

# Reference bounding-box areas (normalised) for distance calibration.
# These are empirical estimates assuming the camera FOV ≈ 110°.
# When a VL53L1X ToF sensor is available these become ground truth.
_REF_AREA_AT_1M   = 0.09   # bbox covers ~9% of frame area at 1 m
_REF_AREA_AT_0_5M = 0.22   # bbox covers ~22% at 0.5 m

def estimate_distance(bbox_norm: Tuple[float, float, float, float]) -> float:
    """
    Heuristic depth estimate from bounding-box area.
    Returns estimated distance in metres.
    Accuracy: ±40% — sufficient for P0/P1/P2 threshold gates.
    Replace with VL53L1X ToF reading when sensor is connected.
    """
    _, _, w, h = bbox_norm
    area = w * h
    if area <= 0:
        return 99.0
    # Inverse-square approximation
    dist = ((_REF_AREA_AT_1M / area) ** 0.5)
    return max(0.1, min(dist, 10.0))


def get_side(cx_norm: float) -> str:
    """Map normalised horizontal centre to left / center / right."""
    if cx_norm < 0.38:
        return "left"
    elif cx_norm > 0.62:
        return "right"
    return "center"


# ─────────────────────────────────────────────
# Height Zone Analysis
# ─────────────────────────────────────────────

def is_head_level(bbox_norm: Tuple[float, float, float, float]) -> bool:
    """
    True if the object's bounding box centre is in the upper 60% of the frame
    (i.e., chest-to-head level in the user's field of view).
    PRD: "Chest-to-head level, on walking trajectory."
    """
    _, cy, _, _ = bbox_norm
    return cy < 0.60

def is_floor_level(bbox_norm: Tuple[float, float, float, float]) -> bool:
    """
    True if the object bottom edge is in the lower 40% of the frame
    (i.e., waist-and-below level).
    """
    _, cy, _, h = bbox_norm
    bottom_y = cy + h / 2
    return bottom_y > 0.55


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────

# Friendly labels map for audio (more natural than raw YOLO class names)
_FRIENDLY_LABELS: dict[str, str] = {
    "dining table":  "table",
    "potted plant":  "plant",
    "cell phone":    "phone",
    "sports ball":   "ball",
    "tennis racket": "racket",
    "hair drier":    "hair dryer",
    "refrigerator":  "fridge",
    "microwave":     "microwave",
    "suitcase":      "suitcase",
    "backpack":      "bag",
    "handbag":       "bag",
    "laptop":        "laptop",
    "tv":            "screen",
    "monitor":       "screen",
}

def _friendly(label: str) -> str:
    return _FRIENDLY_LABELS.get(label.lower(), label.lower())


def build_prompt(priority: HazardPriority, label: str, side: str) -> str:
    """
    Build concise body-anchor prompt per PRD §4.2 grammar:
        [Action / Body Anchor] + [Object Identity]
    """
    fl = _friendly(label)
    if priority == HazardPriority.P0:
        return f"Watch head — {fl}"
    elif priority == HazardPriority.P1:
        if side in ("left", "right"):
            return f"Watch {side} — {fl}"
        return f"Watch steps — {fl}"
    elif priority == HazardPriority.P2:
        if fl in ("cat", "dog", "bird"):
            return "Pet ahead"
        return "Person passing"
    return ""  # NORMAL → silence


# ─────────────────────────────────────────────
# Main Classifier
# ─────────────────────────────────────────────

def classify(
    detections: list[dict],
    frame_shape: Tuple[int, int],
    conf_threshold: float = 0.40,
    p0_dist_threshold: float = 1.2,
    p1_dist_near: float = 0.8,
    p1_dist_far: float = 1.8,
    p2_dist_threshold: float = 2.5,
) -> List[Hazard]:
    """
    Classify YOLO detections into HomeSight hazards.

    Args:
        detections: List of dicts from YOLO, each containing:
            {
              "name": str,        # COCO class name
              "confidence": float,
              "box": {            # pixel coordinates
                "x1": float, "y1": float,
                "x2": float, "y2": float
              }
            }
        frame_shape: (height, width) of the frame in pixels.
        conf_threshold: minimum YOLO confidence to process.
        p0_dist_threshold: distance limit (m) for P0 alerts.
        p1_dist_near / p1_dist_far: distance range for P1 alerts.
        p2_dist_threshold: distance limit (m) for P2 alerts.

    Returns:
        List of Hazard objects sorted by priority descending (P0 first).
        Empty list means SAFE / MUTE — no audio output.
    """
    frame_h, frame_w = frame_shape
    hazards: List[Hazard] = []

    for det in detections:
        conf = float(det.get("confidence", 0))
        if conf < conf_threshold:
            continue

        raw_label = str(det.get("name", "")).lower().strip()
        box = det.get("box", {})

        # Normalise bbox to [0,1]
        x1 = float(box.get("x1", 0)) / frame_w
        y1 = float(box.get("y1", 0)) / frame_h
        x2 = float(box.get("x2", frame_w)) / frame_w
        y2 = float(box.get("y2", frame_h)) / frame_h
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        bbox_norm = (cx, cy, w, h)

        dist = estimate_distance(bbox_norm)
        side = get_side(cx)

        # Skip if in safe class list
        if raw_label in _SAFE_CLASSES:
            continue

        priority = _resolve_priority(
            raw_label, bbox_norm, dist,
            p0_dist_threshold, p1_dist_near, p1_dist_far, p2_dist_threshold
        )

        if priority == HazardPriority.NORMAL:
            continue

        prompt = build_prompt(priority, raw_label, side)
        if not prompt:
            continue

        hazards.append(Hazard(
            priority=priority,
            label=raw_label,
            prompt=prompt,
            side=side,
            conf=conf,
            bbox_norm=bbox_norm,
            est_dist_m=dist,
        ))

    # Sort: highest priority first, then closest distance
    hazards.sort(key=lambda h: (-h.priority.value, h.est_dist_m or 99.0))

    # De-duplicate: keep only the most critical hazard per prompt to prevent spam
    seen_prompts: set[str] = set()
    unique: List[Hazard] = []
    for h in hazards:
        if h.prompt not in seen_prompts:
            seen_prompts.add(h.prompt)
            unique.append(h)

    return unique


def _resolve_priority(
    label: str,
    bbox_norm: Tuple[float, float, float, float],
    dist: float,
    p0_dist: float,
    p1_near: float,
    p1_far: float,
    p2_dist: float,
) -> HazardPriority:
    """Determine hazard priority for a single detection."""

    # ── P2: Dynamic objects ALWAYS win over P1, regardless of floor position ──
    # A person 1.7m away must never degrade to a P1 floor-obstacle alert.
    # Extended range: if someone is in the COCO dynamic class list, flag them
    # up to p2_dist (default 2.5m) — further away = earlier warning.
    if label in _P2_CLASSES:
        if dist < p2_dist:
            return HazardPriority.P2
        # Beyond p2_dist → NORMAL (they're far enough, don't alert)
        return HazardPriority.NORMAL

    # ── P0: Head-level hazard (door, cabinet, etc.) ───────────────────────────
    is_p0_class   = label in _P0_CLASSES
    is_p0_partial = any(p in label for p in ["door", "cabinet", "shelf", "lamp", "light", "fixture"])
    if (is_p0_class or is_p0_partial) and dist < p0_dist and is_head_level(bbox_norm):
        return HazardPriority.P0

    # ── P1: Floor obstacle (chair, fan, box …) ────────────────────────────────
    is_p1_class   = label in _P1_CLASSES
    is_p1_partial = any(p in label for p in _P1_PARTIALS)
    if (is_p1_class or is_p1_partial) and p1_near <= dist <= p1_far and is_floor_level(bbox_norm):
        return HazardPriority.P1

    # Catch-all: unknown object, floor level, within range → P1 (not for P2 classes)
    if dist < p1_far and is_floor_level(bbox_norm) and label not in _SAFE_CLASSES:
        return HazardPriority.P1

    return HazardPriority.NORMAL
