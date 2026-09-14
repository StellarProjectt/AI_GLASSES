#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HomeSight — Latency Benchmark
==============================
Measures end-to-end latency of the HomeSight Python server.
Targets PRD §5.1: total ≤ 250 ms (P99 hard cap: 300 ms).

Usage:
    # Server must be running first:
    # python server.py

    python benchmark_latency.py
    python benchmark_latency.py --frames 100 --server http://192.168.0.102:5001
    python benchmark_latency.py --endpoint analyze --verbose
"""

import argparse
import io
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─────────────────────────────────────────────
# Synthetic test image generator
# ─────────────────────────────────────────────

def create_test_jpeg(width=640, height=480, size_kb_approx=None) -> bytes:
    """
    Generate a synthetic JPEG for benchmarking.
    If PIL not available, returns a minimal valid JPEG stub.
    """
    if PIL_AVAILABLE:
        img = Image.new("RGB", (width, height), color=(40, 40, 60))
        draw = ImageDraw.Draw(img)
        # Draw simple shapes to simulate a scene
        draw.rectangle([50, 60, 300, 280], fill=(80, 60, 40))   # "furniture"
        draw.rectangle([320, 200, 580, 450], fill=(30, 70, 30)) # "plant"
        draw.ellipse([240, 100, 380, 200], fill=(200, 160, 100)) # "light"
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    # Minimal JPEG stub (2×2 pixels, valid JPEG header)
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x02,
        0x00, 0x02, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB,
        0x25, 0xA4, 0xFF, 0xD9
    ])


# ─────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────

def post_multipart(url: str, jpeg_bytes: bytes, timeout: float = 5.0):
    """Send a multipart POST with a JPEG image. Returns (response_bytes, latency_ms)."""
    boundary = b"BenchmarkBoundary42"
    body  = b"--" + boundary + b"\r\n"
    body += b'Content-Disposition: form-data; name="image"; filename="bench.jpg"\r\n'
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += jpeg_bytes
    body += b"\r\n--" + boundary + b"--\r\n"

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    latency_ms = (time.perf_counter() - t0) * 1000
    return data, latency_ms


# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

def run_benchmark(server: str, endpoint: str, n_frames: int, verbose: bool):
    url = server.rstrip("/") + "/" + endpoint.lstrip("/")

    # Health check first
    try:
        with urllib.request.urlopen(server + "/health", timeout=3) as r:
            health = json.loads(r.read())
        print(f"  Server: {health.get('version', '?')}  "
              f"| Model: {health.get('model', '?')}  "
              f"| Audio cache: {health.get('audio_cache', '?')}")
    except Exception as e:
        print(f"  ❌ Cannot reach server at {server}: {e}")
        sys.exit(1)

    jpeg_bytes = create_test_jpeg()
    print(f"  Test image: {len(jpeg_bytes)} bytes  "
          f"| Frames: {n_frames}  "
          f"| Endpoint: {endpoint}")
    print()

    latencies    = []
    server_times = []
    alerts_found = 0
    errors       = 0

    print(f"  {'Frame':>6}  {'Total ms':>9}  {'Server ms':>10}  {'Alert':>7}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*7}")

    for i in range(1, n_frames + 1):
        try:
            raw, lat_ms = post_multipart(url, jpeg_bytes, timeout=5.0)
            response = json.loads(raw)

            server_elapsed = response.get("elapsed", 0) * 1000  # s → ms
            is_alert       = response.get("alert", False)
            if is_alert:
                alerts_found += 1

            latencies.append(lat_ms)
            server_times.append(server_elapsed)

            if verbose or i % 10 == 0:
                alert_str = f"{'YES (' + response.get('priority','?') + ')':>7}" if is_alert else f"{'no':>7}"
                print(f"  {i:>6}  {lat_ms:>9.1f}  {server_elapsed:>10.1f}  {alert_str}")

        except urllib.error.URLError as e:
            errors += 1
            print(f"  {i:>6}  [ERROR: {str(e)[:50]}]")
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  {i:>6}  [EXCEPTION: {str(e)[:50]}]")

    if not latencies:
        print("No successful requests. Exiting.")
        sys.exit(1)

    # ─── Statistics ───────────────────────────────────────────────────────────
    lat_sorted   = sorted(latencies)
    srv_sorted   = sorted(server_times)

    p50  = statistics.median(latencies)
    p90  = lat_sorted[int(len(lat_sorted) * 0.90)]
    p99  = lat_sorted[int(len(lat_sorted) * 0.99)]
    mean = statistics.mean(latencies)
    stdev= statistics.stdev(latencies) if len(latencies) > 1 else 0
    minl = min(latencies)
    maxl = max(latencies)

    srv_p50 = statistics.median(server_times)
    srv_p99 = srv_sorted[int(len(srv_sorted) * 0.99)]

    # PRD targets
    prd_p99_ok = p99 < 300
    prd_mean_ok= mean < 250
    alert_rate = alerts_found / len(latencies) * 100

    print()
    print("  ─" * 32)
    print(f"  {'LATENCY REPORT':^60}")
    print("  ─" * 32)
    print()
    print(f"  Frames processed  : {len(latencies)} / {n_frames}  (errors: {errors})")
    print()
    print(f"  End-to-End (total round-trip, ms)")
    print(f"    Min    : {minl:>8.1f} ms")
    print(f"    Mean   : {mean:>8.1f} ms")
    print(f"    Stdev  : {stdev:>8.1f} ms")
    print(f"    Max    : {maxl:>8.1f} ms")
    print(f"    P50    : {p50:>8.1f} ms")
    print(f"    P90    : {p90:>8.1f} ms")
    print(f"    P99    : {p99:>8.1f} ms  {'✅ < 300ms' if prd_p99_ok else '❌ EXCEEDS 300ms!'}")
    print()
    print(f"  Server-side only (ms)")
    print(f"    P50    : {srv_p50:>8.1f} ms")
    print(f"    P99    : {srv_p99:>8.1f} ms")
    print()
    print(f"  Alerts detected   : {alerts_found} ({alert_rate:.1f}%)")
    print()

    # PRD §6 KPI summary
    print("  PRD KPI Summary:")
    print(f"    P99 < 300 ms       : {'✅ PASS' if prd_p99_ok  else '❌ FAIL'} ({p99:.1f} ms)")
    print(f"    Mean < 250 ms      : {'✅ PASS' if prd_mean_ok else '❌ FAIL'} ({mean:.1f} ms)")
    print(f"    False alarm rate   : {alert_rate:.1f}%  (target < 8% on neutral scenes)")
    print()

    return 0 if prd_p99_ok else 1


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HomeSight server latency benchmark (PRD §5.1 / §6)"
    )
    parser.add_argument("--server",   default="http://localhost:5001",
                        help="Server base URL (default: http://localhost:5001)")
    parser.add_argument("--endpoint", default="analyze",
                        choices=["analyze", "describe"],
                        help="Endpoint to benchmark (default: analyze)")
    parser.add_argument("--frames",   type=int, default=50,
                        help="Number of frames to send (default: 50)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print every frame result")
    args = parser.parse_args()

    print()
    print("=" * 64)
    print("  HomeSight — Latency Benchmark")
    print("  PRD §5.1 target: P99 ≤ 300 ms  |  Mean ≤ 250 ms")
    print("=" * 64)
    print()

    exit_code = run_benchmark(args.server, args.endpoint, args.frames, args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
