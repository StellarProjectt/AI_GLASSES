#!/usr/bin/env python3
"""
HomeSight — YOLOv8n Model Downloader
=====================================
Downloads the YOLOv8n ONNX model for offline local inference.
Run once before starting the server:

    python models/download_model.py

The model (~6 MB) is saved to python_server/models/yolov8n.onnx
"""
import os, sys, urllib.request, hashlib
from pathlib import Path

MODEL_DIR  = Path(__file__).parent
ONNX_PATH  = MODEL_DIR / "yolov8n.onnx"
# Official Ultralytics release asset
ONNX_URL   = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
PT_PATH    = MODEL_DIR / "yolov8n.pt"

def download_pt():
    print(f"Downloading YOLOv8n weights → {PT_PATH} ...")
    def progress(count, block, total):
        done = count * block
        pct  = min(100, int(done * 100 / total)) if total > 0 else 0
        bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}% ({done//1024} KB)", end="", flush=True)
    urllib.request.urlretrieve(ONNX_URL, PT_PATH, reporthook=progress)
    print()

def export_to_onnx():
    print("Exporting YOLOv8n → ONNX (requires ultralytics) ...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(PT_PATH))
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
        # ultralytics saves alongside the .pt file
        exported = PT_PATH.parent / "yolov8n.onnx"
        if exported.exists():
            exported.rename(ONNX_PATH)
            print(f"  Saved: {ONNX_PATH}")
        else:
            print("  Warning: ONNX file not found at expected path, check ultralytics output.")
    except ImportError:
        print("  ultralytics not installed — install it with: pip install ultralytics")
        sys.exit(1)

if __name__ == "__main__":
    if ONNX_PATH.exists():
        print(f"Model already present: {ONNX_PATH} ({ONNX_PATH.stat().st_size // 1024} KB)")
        sys.exit(0)
    download_pt()
    export_to_onnx()
    print("\nDone! You can now start the server.")
