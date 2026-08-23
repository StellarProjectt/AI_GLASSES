# -*- coding: utf-8 -*-
"""
Test script for AI Glasses Python Server
=========================================
Tests:
  1. /health - check server ready
  2. /analyze - send test image, check danger detection
  3. /describe - send test image, get description
  4. /beep - download beep WAV
  5. verify WAV format correctness

Usage:
  1. Run server.py first: python server.py
  2. Run test: python test_server.py
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
import base64
import io
import time
import struct
from PIL import Image

SERVER_URL = "http://localhost:5001"

# สร้างภาพทดสอบ JPEG (ภาพง่ายๆ มีสีเขียว ไม่มีอันตราย)
def create_test_image(width=320, height=240, color=(100, 150, 80)):
    """สร้างภาพ JPEG ทดสอบ"""
    img = Image.new("RGB", (width, height), color)
    # วาดรูปสี่เหลี่ยมเพื่อให้มีรายละเอียดบ้าง
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 270, 190], fill=(200, 180, 100))  # สี่เหลี่ยมเหมือนโต๊ะ
    draw.rectangle([100, 100, 130, 200], fill=(80, 60, 40))     # ขาโต๊ะ
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return buf.getvalue()


def check_wav_format(data):
    """ตรวจสอบว่า data เป็น WAV format ถูกต้อง"""
    if len(data) < 44:
        return False, "Data too short for WAV header"
    
    # Check RIFF header
    if data[:4] != b'RIFF':
        return False, f"Not RIFF (got: {data[:4]})"
    
    if data[8:12] != b'WAVE':
        return False, f"Not WAVE (got: {data[8:12]})"
    
    # Parse fmt chunk
    audio_format = struct.unpack('<H', data[20:22])[0]
    channels = struct.unpack('<H', data[22:24])[0]
    sample_rate = struct.unpack('<I', data[24:28])[0]
    bits_per_sample = struct.unpack('<H', data[34:36])[0]
    
    info = f"WAV: {sample_rate}Hz, {bits_per_sample}bit, {channels}ch, PCM={audio_format==1}"
    return True, info


def test_health():
    """Test 1: Health check"""
    print("\n" + "="*60)
    print("TEST 1: GET /health")
    print("="*60)
    
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        data = resp.json()
        for k, v in data.items():
            print(f"  {k}: {v}")
        
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        assert data["status"] == "ok", f"Expected status=ok, got {data['status']}"
        
        print("  ✅ PASS")
        return True
    except requests.ConnectionError:
        print("  ❌ FAIL — ไม่สามารถเชื่อมต่อ server ได้")
        print(f"     กรุณารัน: python server.py ก่อน")
        return False
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        return False


def test_analyze():
    """Test 2: Auto danger detection"""
    print("\n" + "="*60)
    print("TEST 2: POST /analyze — ตรวจจับอันตราย")
    print("="*60)
    
    try:
        jpeg_data = create_test_image()
        print(f"  Sending test image: {len(jpeg_data)} bytes")
        
        start = time.time()
        resp = requests.post(
            f"{SERVER_URL}/analyze",
            files={"image": ("test.jpg", jpeg_data, "image/jpeg")},
            timeout=30
        )
        elapsed = time.time() - start
        
        print(f"  Status: {resp.status_code}")
        print(f"  Response time: {elapsed:.1f}s")
        
        data = resp.json()
        print(f"  alert: {data.get('alert')}")
        print(f"  message: {data.get('message', '(none)')}")
        print(f"  req_id: {data.get('req_id', '(none)')}")
        print(f"  server elapsed: {data.get('elapsed', '?')}s")
        
        # ถ้ามี audio ให้ตรวจสอบ format
        if "audio_base64" in data:
            audio_bytes = base64.b64decode(data["audio_base64"])
            audio_fmt = data.get("audio_format", "unknown")
            audio_size = data.get("audio_size", len(audio_bytes))
            print(f"  audio_format: {audio_fmt}")
            print(f"  audio_size: {audio_size} bytes ({audio_size/1024:.1f} KB)")
            
            if audio_fmt == "wav":
                is_wav, wav_info = check_wav_format(audio_bytes)
                print(f"  WAV check: {wav_info}")
                if is_wav:
                    print("  ✅ Audio is valid WAV format!")
                else:
                    print("  ⚠️  Audio is NOT valid WAV")
            
            # บันทึกไฟล์เสียงเพื่อทดสอบฟัง
            ext = "wav" if audio_fmt == "wav" else "mp3"
            with open(f"test_analyze_output.{ext}", "wb") as f:
                f.write(audio_bytes)
            print(f"  💾 Saved: test_analyze_output.{ext}")
        else:
            print("  (No audio — ภาพไม่มีอันตราย)")
        
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        print("  ✅ PASS")
        return True
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        import traceback
        traceback.print_exc()
        return False


def test_describe():
    """Test 3: Scene description"""
    print("\n" + "="*60)
    print("TEST 3: POST /describe — อธิบายภาพ")
    print("="*60)
    
    try:
        jpeg_data = create_test_image()
        print(f"  Sending test image: {len(jpeg_data)} bytes")
        
        start = time.time()
        resp = requests.post(
            f"{SERVER_URL}/describe",
            files={"image": ("test.jpg", jpeg_data, "image/jpeg")},
            timeout=30
        )
        elapsed = time.time() - start
        
        print(f"  Status: {resp.status_code}")
        print(f"  Response time: {elapsed:.1f}s")
        
        data = resp.json()
        print(f"  description: \"{data.get('description', '(none)')}\"")
        print(f"  req_id: {data.get('req_id', '(none)')}")
        print(f"  server elapsed: {data.get('elapsed', '?')}s")
        
        # ตรวจ audio
        if "audio_base64" in data:
            audio_bytes = base64.b64decode(data["audio_base64"])
            audio_fmt = data.get("audio_format", "unknown")
            audio_size = data.get("audio_size", len(audio_bytes))
            print(f"  audio_format: {audio_fmt}")
            print(f"  audio_size: {audio_size} bytes ({audio_size/1024:.1f} KB)")
            
            if audio_fmt == "wav":
                is_wav, wav_info = check_wav_format(audio_bytes)
                print(f"  WAV check: {wav_info}")
                if is_wav:
                    print("  ✅ Audio is valid WAV format!")
                else:
                    print("  ⚠️  Audio is NOT valid WAV")
            
            # บันทึกไฟล์เสียงเพื่อทดสอบฟัง
            ext = "wav" if audio_fmt == "wav" else "mp3"
            with open(f"test_describe_output.{ext}", "wb") as f:
                f.write(audio_bytes)
            print(f"  💾 Saved: test_describe_output.{ext}")
        
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        assert "description" in data, "No 'description' in response"
        assert len(data["description"]) > 0, "Description is empty"
        print("  ✅ PASS")
        return True
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        import traceback
        traceback.print_exc()
        return False


def test_beep():
    """Test 4: Beep WAV download"""
    print("\n" + "="*60)
    print("TEST 4: GET /beep — ดาวน์โหลดเสียง beep")
    print("="*60)
    
    try:
        resp = requests.get(f"{SERVER_URL}/beep", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
        print(f"  Size: {len(resp.content)} bytes ({len(resp.content)/1024:.1f} KB)")
        
        is_wav, wav_info = check_wav_format(resp.content)
        print(f"  WAV check: {wav_info}")
        
        with open("test_beep_output.wav", "wb") as f:
            f.write(resp.content)
        print(f"  💾 Saved: test_beep_output.wav")
        
        assert resp.status_code == 200
        assert is_wav, "Beep is not valid WAV"
        print("  ✅ PASS")
        return True
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        return False


def test_error_handling():
    """Test 5: Error handling — ส่ง request ผิด format"""
    print("\n" + "="*60)
    print("TEST 5: Error Handling — ส่ง request ไม่ถูกต้อง")
    print("="*60)
    
    try:
        # Test: POST /analyze without image
        resp = requests.post(f"{SERVER_URL}/analyze", timeout=5)
        print(f"  No image: Status={resp.status_code} (expected 400)")
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
        
        # Test: GET /nonexistent
        resp = requests.get(f"{SERVER_URL}/nonexistent", timeout=5)
        print(f"  Not found: Status={resp.status_code} (expected 404)")
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"
        
        print("  ✅ PASS")
        return True
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        return False


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     AI Glasses — Python Server Test Suite               ║")
    print("║     ทดสอบ server ก่อนใช้กับ ESP32                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Target: {SERVER_URL}")
    
    results = {}
    
    # Test 1: Health
    results["health"] = test_health()
    if not results["health"]:
        print("\n⛔ Server ไม่พร้อม — ยกเลิกการทดสอบที่เหลือ")
        print(f"   กรุณารัน: python server.py ก่อนแล้วรัน test ใหม่")
        sys.exit(1)
    
    # Test 2: Analyze
    results["analyze"] = test_analyze()
    
    # Test 3: Describe
    results["describe"] = test_describe()
    
    # Test 4: Beep
    results["beep"] = test_beep()
    
    # Test 5: Error handling
    results["errors"] = test_error_handling()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  Result: {passed}/{total} passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Server พร้อมใช้กับ ESP32 แล้ว")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed — กรุณาตรวจสอบ log ด้านบน")
    print()
