# -*- coding: utf-8 -*-
"""
HomeSight — Hazard Classifier Unit Tests
==========================================
Run: python -m pytest test_hazard_classifier.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hazard_classifier import (
    classify, HazardPriority, Hazard,
    estimate_distance, get_side, is_head_level, is_floor_level,
    build_prompt,
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
FRAME = (480, 640)  # (height, width)

def make_det(name, x1, y1, x2, y2, conf=0.85):
    """Construct a detection dict in YOLO format."""
    return {
        "name": name,
        "confidence": conf,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }

def det_at(name, cx_frac, cy_frac, w_frac, h_frac, conf=0.85):
    """Create detection from normalised centre + size fractions."""
    fh, fw = FRAME
    cx, cy = cx_frac * fw, cy_frac * fh
    w,  h  = w_frac  * fw, h_frac  * fh
    return make_det(name, cx - w/2, cy - h/2, cx + w/2, cy + h/2, conf)


# ─────────────────────────────────────────────
# Section 1 — Silence by default (PRD Principle #1)
# ─────────────────────────────────────────────

class TestSilenceByDefault:
    def test_empty_detections_returns_no_hazards(self):
        result = classify([], FRAME)
        assert result == [], "Empty frame must produce zero alerts"

    def test_safe_classes_are_suppressed(self):
        dets = [
            det_at("wall",   0.5, 0.5, 0.8, 0.9),
            det_at("couch",  0.5, 0.6, 0.6, 0.4),
            det_at("ceiling",0.5, 0.1, 1.0, 0.2),
        ]
        result = classify(dets, FRAME)
        assert result == [], "Safe architecture must be totally silent"

    def test_low_confidence_suppressed(self):
        dets = [det_at("chair", 0.5, 0.8, 0.2, 0.3, conf=0.25)]
        result = classify(dets, FRAME)
        assert result == [], "Detections below confidence threshold must be ignored"

    def test_distant_objects_are_silent(self):
        # Very small bbox → large estimated distance → no alert
        dets = [det_at("chair", 0.5, 0.9, 0.02, 0.03)]  # tiny bbox = very far
        result = classify(dets, FRAME)
        # Should be empty or NORMAL only
        for h in result:
            assert h.priority == HazardPriority.NORMAL, \
                "Distant objects must not produce P0/P1/P2 alerts"


# ─────────────────────────────────────────────
# Section 2 — P0 Head Hazard (PRD §4.1)
# ─────────────────────────────────────────────

class TestP0HeadHazard:
    """
    Trigger criteria: Distance < 1.2 m, chest-to-head level (cy < 0.60),
    on walking trajectory.
    """

    def _make_p0_det(self, name="cabinet"):
        # Large bbox (close), high in frame (head level)
        return det_at(name, 0.5, 0.3, 0.35, 0.40)  # cy=0.30 → head level

    def test_open_cabinet_triggers_p0(self):
        result = classify([self._make_p0_det("cabinet")], FRAME)
        assert any(h.priority == HazardPriority.P0 for h in result), \
            "Cabinet at head height within range must be P0"

    def test_door_triggers_p0(self):
        result = classify([self._make_p0_det("door")], FRAME)
        assert any(h.priority == HazardPriority.P0 for h in result), \
            "Door at head height within range must be P0"

    def test_p0_prompt_grammar(self):
        result = classify([self._make_p0_det("cabinet")], FRAME)
        p0 = next(h for h in result if h.priority == HazardPriority.P0)
        assert p0.prompt.startswith("Watch head —"), \
            f"P0 prompt must start with 'Watch head —', got: '{p0.prompt}'"

    def test_p0_earcon_is_high(self):
        result = classify([self._make_p0_det()], FRAME)
        p0 = next(h for h in result if h.priority == HazardPriority.P0)
        assert p0.earcon_code == "HIGH", "P0 must use HIGH earcon"

    def test_cabinet_below_waist_is_not_p0(self):
        # Same cabinet but at floor level (cy = 0.85) → should be P1 not P0
        det = det_at("cabinet", 0.5, 0.85, 0.35, 0.25)
        result = classify([det], FRAME)
        p0_hazards = [h for h in result if h.priority == HazardPriority.P0]
        assert len(p0_hazards) == 0, \
            "Cabinet at floor level must not trigger P0 head hazard"


# ─────────────────────────────────────────────
# Section 3 — P1 Floor Obstacle (PRD §4.1)
# ─────────────────────────────────────────────

class TestP1FloorObstacle:
    """
    Trigger criteria: Distance 0.8–1.8 m, below waist level,
    on floor plane.
    """

    def _make_p1_det(self, name="chair"):
        # Medium-size bbox (1–1.5 m range), bottom of frame
        return det_at(name, 0.5, 0.78, 0.20, 0.30)  # cy+h/2 = 0.93 → floor

    def test_chair_in_path_triggers_p1(self):
        result = classify([self._make_p1_det("chair")], FRAME)
        assert any(h.priority == HazardPriority.P1 for h in result), \
            "Chair on floor within range must be P1"

    def test_floor_fan_triggers_p1(self):
        result = classify([self._make_p1_det("fan")], FRAME)
        assert any(h.priority == HazardPriority.P1 for h in result), \
            "Fan on floor must be P1 (partial match)"

    def test_p1_center_prompt(self):
        det = det_at("chair", 0.5, 0.78, 0.20, 0.30)  # center
        result = classify([det], FRAME)
        p1 = next((h for h in result if h.priority == HazardPriority.P1), None)
        assert p1 is not None
        assert "Watch steps" in p1.prompt or "Watch" in p1.prompt, \
            f"P1 center prompt incorrect: '{p1.prompt}'"

    def test_p1_left_side_prompt(self):
        det = det_at("chair", 0.15, 0.78, 0.20, 0.30)  # left
        result = classify([det], FRAME)
        p1 = next((h for h in result if h.priority == HazardPriority.P1), None)
        assert p1 is not None
        assert "left" in p1.prompt.lower(), \
            f"P1 left-side prompt must contain 'left', got: '{p1.prompt}'"

    def test_p1_earcon_is_low(self):
        result = classify([self._make_p1_det()], FRAME)
        p1 = next((h for h in result if h.priority == HazardPriority.P1), None)
        assert p1 is not None
        assert p1.earcon_code == "LOW", "P1 must use LOW earcon"


# ─────────────────────────────────────────────
# Section 4 — P2 Dynamic Objects (PRD §4.1)
# ─────────────────────────────────────────────

class TestP2Dynamic:
    def _make_p2_det(self, name="person"):
        return det_at(name, 0.5, 0.5, 0.25, 0.60)  # medium, center

    def test_person_triggers_p2(self):
        result = classify([self._make_p2_det("person")], FRAME)
        assert any(h.priority == HazardPriority.P2 for h in result)

    def test_dog_triggers_p2(self):
        result = classify([self._make_p2_det("dog")], FRAME)
        assert any(h.priority == HazardPriority.P2 for h in result)

    def test_person_prompt(self):
        result = classify([self._make_p2_det("person")], FRAME)
        p2 = next((h for h in result if h.priority == HazardPriority.P2), None)
        assert p2 is not None
        assert p2.prompt == "Person passing", f"Got: '{p2.prompt}'"

    def test_pet_prompt(self):
        result = classify([self._make_p2_det("cat")], FRAME)
        p2 = next((h for h in result if h.priority == HazardPriority.P2), None)
        assert p2 is not None
        assert p2.prompt == "Pet ahead", f"Got: '{p2.prompt}'"

    def test_p2_earcon_is_dir(self):
        result = classify([self._make_p2_det()], FRAME)
        p2 = next((h for h in result if h.priority == HazardPriority.P2), None)
        assert p2 is not None
        assert p2.earcon_code == "DIR", "P2 must use DIR earcon"


# ─────────────────────────────────────────────
# Section 5 — Priority Ordering (P0 first)
# ─────────────────────────────────────────────

class TestPriorityOrdering:
    def test_p0_before_p1_in_results(self):
        dets = [
            det_at("chair",   0.5, 0.80, 0.20, 0.30),   # P1
            det_at("cabinet", 0.5, 0.25, 0.35, 0.40),   # P0 (head level, close)
        ]
        result = classify(dets, FRAME)
        hazard_priorities = [h.priority for h in result]
        p0_idx = next((i for i, p in enumerate(hazard_priorities) if p == HazardPriority.P0), None)
        p1_idx = next((i for i, p in enumerate(hazard_priorities) if p == HazardPriority.P1), None)
        assert p0_idx is not None
        assert p1_idx is not None
        assert p0_idx < p1_idx, "P0 must appear before P1 in sorted results"

    def test_deduplication(self):
        # Two chairs in the same location → only one alert
        dets = [
            det_at("chair", 0.50, 0.78, 0.20, 0.30),
            det_at("chair", 0.55, 0.80, 0.18, 0.28),
        ]
        result = classify(dets, FRAME)
        chair_alerts = [h for h in result if "chair" in h.label]
        # Should deduplicate to unique prompts
        prompts = [h.prompt for h in chair_alerts]
        assert len(set(prompts)) == len(prompts), "Duplicate prompts must be deduplicated"


# ─────────────────────────────────────────────
# Section 6 — Side Detection
# ─────────────────────────────────────────────

class TestSideDetection:
    def test_left_object(self):
        side = get_side(0.15)
        assert side == "left"

    def test_right_object(self):
        side = get_side(0.85)
        assert side == "right"

    def test_center_object(self):
        side = get_side(0.50)
        assert side == "center"


# ─────────────────────────────────────────────
# Section 7 — Distance Heuristic
# ─────────────────────────────────────────────

class TestDistanceEstimation:
    def test_large_bbox_is_close(self):
        dist = estimate_distance((0.5, 0.5, 0.4, 0.5))  # 20% area → ~close
        assert dist < 1.5, f"Large bbox should be close, got {dist:.2f} m"

    def test_small_bbox_is_far(self):
        dist = estimate_distance((0.5, 0.5, 0.03, 0.04))  # tiny bbox → far
        assert dist > 2.0, f"Small bbox should be far, got {dist:.2f} m"

    def test_zero_area_returns_max(self):
        dist = estimate_distance((0.5, 0.5, 0.0, 0.0))
        assert dist >= 9.0


# ─────────────────────────────────────────────
# Section 8 — PRD Scenarios (integration)
# ─────────────────────────────────────────────

class TestPRDScenarios:
    def test_scenario_a_head_cabinet(self):
        """Scenario A: Upper cabinet door open at eye level."""
        dets = [det_at("cabinet", 0.5, 0.28, 0.38, 0.45)]
        result = classify(dets, FRAME)
        assert any(h.priority == HazardPriority.P0 for h in result), \
            "Scenario A must produce a P0 head hazard alert"
        p0 = next(h for h in result if h.priority == HazardPriority.P0)
        assert "Watch head" in p0.prompt

    def test_scenario_b_floor_fan(self):
        """Scenario B: Floor fan in hallway."""
        dets = [det_at("fan", 0.5, 0.80, 0.22, 0.30)]
        result = classify(dets, FRAME)
        assert any(h.priority == HazardPriority.P1 for h in result), \
            "Scenario B must produce a P1 floor obstacle alert"
        p1 = next(h for h in result if h.priority == HazardPriority.P1)
        assert "Watch" in p1.prompt

    def test_scenario_c_clear_doorway(self):
        """Scenario C: Clear door frame — system must be silent."""
        # No detections, or only safe architecture
        result = classify([], FRAME)
        assert result == [], "Scenario C must produce zero alerts (silence)"
