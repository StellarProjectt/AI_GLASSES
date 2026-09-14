#!/usr/bin/env python3
"""
HomeSight — YOLOv8 Model Downloader
=====================================
Downloads a YOLOv8 ONNX model for offline local inference.
Run once before starting the server:

    python models/download_model.py

Default: yolov8s ("small", ~22 MB) — meaningfully more accurate than the
nano model at a CPU-latency cost that's still fine on a desktop/laptop.
For slower hardware (or to match server.py's env-var override), request
the nano model instead:

    MODEL_SIZE=n python models/download_model.py
"""
import os, sys, urllib.request, hashlib
from pathlib import Path

MODEL_SIZE = os.environ.get("MODEL_SIZE", "s").strip().lower()
if MODEL_SIZE not in ("n", "s", "m"):
    print(f"Unknown MODEL_SIZE '{MODEL_SIZE}' — expected n (nano), s (small), or m (medium).")
    sys.exit(1)

MODEL_NAME = f"yolov8{MODEL_SIZE}"
MODEL_DIR  = Path(__file__).parent
ONNX_PATH  = MODEL_DIR / f"{MODEL_NAME}.onnx"
PT_PATH    = MODEL_DIR / f"{MODEL_NAME}.pt"
# Official Ultralytics release asset
ONNX_URL   = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{MODEL_NAME}.pt"

def download_pt():
    print(f"Downloading {MODEL_NAME} weights → {PT_PATH} ...")
    def progress(count, block, total):
        done = count * block
        pct  = min(100, int(done * 100 / total)) if total > 0 else 0
        bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}% ({done//1024} KB)", end="", flush=True)
    urllib.request.urlretrieve(ONNX_URL, PT_PATH, reporthook=progress)
    print()

def export_to_onnx():
    print(f"Exporting {MODEL_NAME} → ONNX (requires ultralytics) ...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(PT_PATH))
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
        # ultralytics saves alongside the .pt file
        exported = PT_PATH.parent / f"{MODEL_NAME}.onnx"
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
    print(f"\nDone! Set MODEL_PATH=models/{MODEL_NAME}.onnx (or MODEL_PATH_PT=models/{MODEL_NAME}.pt) "
          f"if this isn't server.py's current default, then start the server.")
