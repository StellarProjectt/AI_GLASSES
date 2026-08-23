# -*- coding: utf-8 -*-
"""
AI Glasses - Python Server v2.0 (Final Production)
====================================================
รับภาพจาก ESP32 -> วิเคราะห์ด้วย Gemini Flash -> สร้างเสียงไทย (gTTS) -> แปลงเป็น WAV PCM -> ส่งกลับ

Endpoints:
  POST /analyze  - ตรวจจับอันตรายอัตโนมัติ (ส่ง JPEG มา, ได้ WAV PCM กลับ ถ้ามีอันตราย)
  POST /describe - อธิบายภาพเมื่อกดปุ่ม (ส่ง JPEG มา, ได้ WAV PCM กลับเสมอ)
  GET  /health   - เช็คว่า server พร้อม
  GET  /beep     - ส่งเสียง beep WAV สำหรับ ESP32

Audio Format ที่ส่งกลับ:
  - WAV PCM: 16kHz, 16-bit, Mono (เล่นผ่าน I2S ที่ ESP32 ได้ทันที)
  - ส่งเป็น base64 encoded ใน JSON field "audio_base64"

Logging:
  - ทุก Request จะมี [REQ-XXXX] ID ติดตาม
  - แสดง Latency breakdown: Gemini API / TTS / WAV Convert / Total
  - แสดง Error stack trace เต็มรูปแบบเมื่อเกิดข้อผิดพลาด

Usage:
  1. ใส่ Gemini API key ใน GEMINI_API_KEY ข้างล่าง (หรือ set environment variable)
  2. pip install -r requirements.txt
  3. python server.py
  4. ESP32 จะส่งภาพมาที่ http://<your-ip>:5001
"""

import os
import sys
import io
import time
import base64
import struct
import subprocess
import traceback
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from PIL import Image
from google import genai
from gtts import gTTS

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# Configuration
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "***REMOVED-LEAKED-GEMINI-KEY***")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5001"))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").lower() == "true"

# WAV output settings for ESP32 I2S
WAV_SAMPLE_RATE = 16000
WAV_BITS = 16
WAV_CHANNELS = 1

# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AI_Glasses")

# ==========================================
# Check ffmpeg availability
# ==========================================
FFMPEG_AVAILABLE = False
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        FFMPEG_AVAILABLE = True
        logger.info("[INIT] ffmpeg found -- WAV conversion enabled")
    else:
        logger.warning("[INIT] ffmpeg found but returned error")
except FileNotFoundError:
    logger.warning("[INIT] ffmpeg NOT found -- audio will be sent as MP3 (ESP32 speaker won't work)")
    logger.warning("[INIT] Install ffmpeg: https://ffmpeg.org/download.html")
except Exception as e:
    logger.warning(f"[INIT] ffmpeg check failed: {e}")

# ==========================================
# Initialize
# ==========================================
app = Flask(__name__)

# Configure Gemini (new google.genai SDK)
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"

logger.info(f"[INIT] Gemini model: {GEMINI_MODEL}")
logger.info(f"[INIT] Audio output: {'WAV PCM (16kHz/16bit/mono)' if FFMPEG_AVAILABLE else 'MP3 (ffmpeg not available)'}")

# Request counter for unique IDs
_request_counter = 0

def get_request_id():
    """Generate a unique short request ID for tracking"""
    global _request_counter
    _request_counter += 1
    return f"REQ-{_request_counter:04d}"


# ==========================================
# Helper: Timer class for latency breakdown
# ==========================================
class LatencyTimer:
    """Track latency of each processing step for debugging"""
    def __init__(self, req_id):
        self.req_id = req_id
        self.start_time = time.time()
        self.steps = []
        self._step_start = None
        self._step_name = None
    
    def begin_step(self, name):
        """Start timing a named step"""
        self._step_name = name
        self._step_start = time.time()
    
    def end_step(self):
        """End the current step and record its duration"""
        if self._step_start and self._step_name:
            elapsed = time.time() - self._step_start
            self.steps.append((self._step_name, elapsed))
            self._step_start = None
            self._step_name = None
    
    def total(self):
        """Total time since timer creation"""
        return time.time() - self.start_time
    
    def summary(self):
        """Return formatted latency breakdown string"""
        parts = []
        for name, elapsed in self.steps:
            parts.append(f"{name}: {elapsed*1000:.0f}ms")
        parts.append(f"TOTAL: {self.total()*1000:.0f}ms")
        return " | ".join(parts)


# ==========================================
# Helper: Convert MP3 to WAV using ffmpeg subprocess
# (Works on Python 3.13 without pydub/audioop)
# ==========================================
def mp3_to_wav_ffmpeg(mp3_bytes):
    """Convert MP3 bytes to WAV PCM (16kHz, 16-bit, mono) using ffmpeg subprocess"""
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-i", "pipe:0",          # Input from stdin
                "-f", "wav",             # Output format: WAV
                "-ar", str(WAV_SAMPLE_RATE),  # Sample rate: 16kHz
                "-ac", str(WAV_CHANNELS),     # Channels: mono
                "-sample_fmt", "s16",    # 16-bit signed
                "-y",                    # Overwrite
                "pipe:1"                 # Output to stdout
            ],
            input=mp3_bytes,
            capture_output=True,
            timeout=30
        )
        
        if proc.returncode != 0:
            logger.error(f"  ffmpeg conversion failed: {proc.stderr.decode('utf-8', errors='replace')[:200]}")
            return None
        
        return proc.stdout
    except subprocess.TimeoutExpired:
        logger.error("  ffmpeg conversion timed out (30s)")
        return None
    except Exception as e:
        logger.error(f"  ffmpeg conversion error: {e}")
        return None


def tts_to_wav_bytes(text, lang="th", timer=None):
    """
    Generate Thai TTS audio and return as WAV bytes (16-bit, 16kHz, mono).
    
    Returns:
        tuple: (audio_bytes, audio_format_string)
        audio_bytes: bytes of the audio file
        audio_format_string: "wav" or "mp3"
    """
    try:
        # Step 1: Generate MP3 via gTTS
        if timer:
            timer.begin_step("gTTS")
        
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_bytes = mp3_buf.getvalue()
        
        if timer:
            timer.end_step()
        
        logger.debug(f"  gTTS generated MP3: {len(mp3_bytes)} bytes ({len(mp3_bytes)/1024:.1f} KB)")
        
        # Step 2: Convert MP3 -> WAV PCM using ffmpeg
        if FFMPEG_AVAILABLE:
            if timer:
                timer.begin_step("MP3->WAV")
            
            wav_bytes = mp3_to_wav_ffmpeg(mp3_bytes)
            
            if timer:
                timer.end_step()
            
            if wav_bytes:
                logger.debug(f"  WAV converted: {len(wav_bytes)} bytes ({len(wav_bytes)/1024:.1f} KB), "
                            f"{WAV_SAMPLE_RATE}Hz/{WAV_BITS}bit/mono")
                return wav_bytes, "wav"
            else:
                logger.warning("  WAV conversion failed -- falling back to MP3")
                return mp3_bytes, "mp3"
        else:
            # Fallback: return MP3 as-is (Web UI can play, ESP32 speaker can't)
            logger.warning("  ffmpeg not available -- returning MP3")
            return mp3_bytes, "mp3"
    
    except Exception as e:
        logger.error(f"  TTS Error: {e}")
        logger.error(traceback.format_exc())
        return None, None


def generate_pcm_beep(frequency=880, duration_ms=200, sample_rate=16000):
    """Generate raw PCM beep for ESP32 I2S speaker (16-bit, mono)"""
    import math
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        envelope = 1.0
        fade_samples = num_samples // 10
        if i < fade_samples:
            envelope = i / fade_samples
        elif i > num_samples - fade_samples:
            envelope = (num_samples - i) / fade_samples
        value = int(math.sin(2 * math.pi * frequency * t) * 16000 * envelope)
        samples.extend(struct.pack('<h', max(-32768, min(32767, value))))
    return bytes(samples)


def create_wav_header(pcm_data, sample_rate=16000, bits=16, channels=1):
    """Wrap raw PCM data in a WAV header"""
    data_size = len(pcm_data)
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    header += struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, channels, 
                          sample_rate, sample_rate * channels * bits // 8,
                          channels * bits // 8, bits)
    header += struct.pack('<4sI', b'data', data_size)
    return header + pcm_data


# ==========================================
# Gemini Analysis Functions
# ==========================================
DANGER_PROMPT = """คุณเป็นระบบช่วยเหลือคนตาบอด ดูภาพนี้แล้วตอบว่ามีอันตรายไหม

กฎ:
1. ถ้ามีอันตราย (หลุม, บันได, สิ่งกีดขวาง, ขอบถนน, รถ, คนเดินสวนมา, น้ำบนพื้น, ประตูกระจก):
   ตอบ: DANGER|คำเตือนสั้นๆ ภาษาไทย (ไม่เกิน 15 คำ)
   ตัวอย่าง: DANGER|ระวัง มีบันไดข้างหน้า ประมาณ 3 ขั้น
   ตัวอย่าง: DANGER|มีรถจอดขวางทางเดินด้านซ้าย

2. ถ้าไม่มีอันตราย:
   ตอบ: SAFE

ตอบแค่บรรทัดเดียว ไม่ต้องอธิบายเพิ่ม"""

DESCRIBE_PROMPT = """คุณเป็นระบบช่วยเหลือคนตาบอด อธิบายสิ่งที่เห็นในภาพให้คนตาบอดฟัง

กฎ:
1. ใช้ภาษาไทย กระชับ ชัดเจน
2. บอกตำแหน่ง (ซ้าย กลาง ขวา ข้างหน้า) 
3. เรียงลำดับจากสิ่งที่ใกล้ที่สุดก่อน
4. ถ้ามีอันตราย ให้เตือนก่อน
5. ความยาวไม่เกิน 3 ประโยค

ตัวอย่าง: ข้างหน้ามีโต๊ะไม้ตรงกลาง มีเก้าอี้ 2 ตัวอยู่ด้านซ้าย ทางเดินด้านขวาว่างสามารถเดินผ่านได้"""


def analyze_danger(image_bytes, timer=None):
    """Send image to Gemini and check for dangers"""
    try:
        if timer:
            timer.begin_step("Gemini")
        
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size
        logger.debug(f"  Image size: {img_width}x{img_height}")
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[DANGER_PROMPT, image],
            config={
                "max_output_tokens": 100,
                "temperature": 0.1,
            }
        )
        result = response.text.strip()
        
        if timer:
            timer.end_step()
        
        logger.info(f"  Gemini result (danger): \"{result}\"")
        
        if result.startswith("DANGER|"):
            warning = result.split("|", 1)[1]
            return True, warning
        return False, ""
    except Exception as e:
        if timer and timer._step_start:
            timer.end_step()
        logger.error(f"  Gemini API Error: {e}")
        logger.error(traceback.format_exc())
        return False, f"ระบบผิดพลาด: {str(e)[:80]}"


def describe_scene(image_bytes, timer=None):
    """Send image to Gemini and get scene description"""
    try:
        if timer:
            timer.begin_step("Gemini")
        
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size
        logger.debug(f"  Image size: {img_width}x{img_height}")
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[DESCRIBE_PROMPT, image],
            config={
                "max_output_tokens": 200,
                "temperature": 0.3,
            }
        )
        result = response.text.strip()
        
        if timer:
            timer.end_step()
        
        logger.info(f"  Gemini result (describe): \"{result}\"")
        return result
    except Exception as e:
        if timer and timer._step_start:
            timer.end_step()
        logger.error(f"  Gemini API Error: {e}")
        logger.error(traceback.format_exc())
        return "ไม่สามารถวิเคราะห์ภาพได้ ลองอีกครั้ง"


# ==========================================
# API Endpoints
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    """Health check -- ESP32 calls this to verify server is reachable"""
    logger.debug("[HEALTH] Health check received")
    return jsonify({
        "status": "ok",
        "model": GEMINI_MODEL,
        "version": "2.0-final",
        "audio_format": "wav" if FFMPEG_AVAILABLE else "mp3",
        "wav_spec": f"{WAV_SAMPLE_RATE}Hz/{WAV_BITS}bit/mono",
        "ffmpeg": FFMPEG_AVAILABLE,
        "uptime": int(time.time() - _server_start_time)
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Auto danger detection endpoint.
    ESP32 sends JPEG image, server checks for dangers.
    
    Returns:
      - If danger: JSON with alert=true + WAV audio of Thai warning (base64)
      - If safe: JSON with alert=false (no audio)
    """
    req_id = get_request_id()
    timer = LatencyTimer(req_id)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[{req_id}] POST /analyze -- Auto Danger Detection")
    logger.info("=" * 60)
    
    if "image" not in request.files:
        logger.error(f"[{req_id}] ERROR: No image file in request")
        return jsonify({"error": "No image provided", "req_id": req_id}), 400
    
    timer.begin_step("Upload")
    image_data = request.files["image"].read()
    timer.end_step()
    
    logger.info(f"[{req_id}] Received image: {len(image_data)} bytes ({len(image_data)/1024:.1f} KB)")
    
    # Analyze with Gemini
    is_danger, warning = analyze_danger(image_data, timer)
    
    if is_danger:
        logger.warning(f"[{req_id}] !! DANGER DETECTED: \"{warning}\"")
        
        # Generate TTS audio
        audio_data, audio_fmt = tts_to_wav_bytes(warning, timer=timer)
        
        result = {
            "alert": True,
            "message": warning,
            "elapsed": round(timer.total(), 2),
            "req_id": req_id
        }
        
        if audio_data:
            timer.begin_step("Base64")
            result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")
            result["audio_format"] = audio_fmt
            result["audio_size"] = len(audio_data)
            timer.end_step()
            logger.info(f"[{req_id}] Audio: {len(audio_data)} bytes ({audio_fmt.upper()})")
        
        logger.info(f"[{req_id}] Latency: {timer.summary()}")
        logger.info(f"[{req_id}] Response sent (DANGER)")
        return jsonify(result)
    else:
        logger.info(f"[{req_id}] SAFE -- No danger detected")
        logger.info(f"[{req_id}] Latency: {timer.summary()}")
        return jsonify({
            "alert": False,
            "elapsed": round(timer.total(), 2),
            "req_id": req_id
        })


@app.route("/describe", methods=["POST"])
def describe():
    """
    Scene description endpoint (triggered by button press).
    ESP32 sends JPEG image, server describes the scene in Thai.
    
    Returns: JSON with description text + WAV audio (base64)
    """
    req_id = get_request_id()
    timer = LatencyTimer(req_id)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[{req_id}] POST /describe -- Scene Description (Button Press)")
    logger.info("=" * 60)
    
    if "image" not in request.files:
        logger.error(f"[{req_id}] ERROR: No image file in request")
        return jsonify({"error": "No image provided", "req_id": req_id}), 400
    
    timer.begin_step("Upload")
    image_data = request.files["image"].read()
    timer.end_step()
    
    logger.info(f"[{req_id}] Received image: {len(image_data)} bytes ({len(image_data)/1024:.1f} KB)")
    
    # Describe with Gemini
    description = describe_scene(image_data, timer)
    
    logger.info(f"[{req_id}] Description: \"{description}\"")
    
    # Generate TTS audio
    audio_data, audio_fmt = tts_to_wav_bytes(description, timer=timer)
    
    result = {
        "description": description,
        "elapsed": round(timer.total(), 2),
        "req_id": req_id
    }
    
    if audio_data:
        timer.begin_step("Base64")
        result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")
        result["audio_format"] = audio_fmt
        result["audio_size"] = len(audio_data)
        timer.end_step()
        logger.info(f"[{req_id}] Audio: {len(audio_data)} bytes ({audio_fmt.upper()})")
    
    logger.info(f"[{req_id}] Latency: {timer.summary()}")
    logger.info(f"[{req_id}] Response sent (DESCRIBE)")
    return jsonify(result)


@app.route("/beep", methods=["GET"])
def beep():
    """Return a simple alert beep WAV for ESP32 I2S speaker"""
    logger.debug("[BEEP] Generating alert beep WAV")
    beep1 = generate_pcm_beep(880, 150)
    silence = b'\x00\x00' * 800  # 50ms silence
    beep2 = generate_pcm_beep(1100, 200)
    pcm = beep1 + silence + beep2
    wav = create_wav_header(pcm)
    
    logger.debug(f"[BEEP] Generated {len(wav)} bytes WAV")
    return send_file(
        io.BytesIO(wav),
        mimetype="audio/wav",
        as_attachment=False
    )


# ==========================================
# Error Handlers
# ==========================================
@app.errorhandler(404)
def not_found(e):
    logger.warning(f"[404] {request.method} {request.path} -- Not Found")
    return jsonify({"error": "Endpoint not found", "path": request.path}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"[500] Internal server error: {e}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500


# ==========================================
# Main
# ==========================================
_server_start_time = time.time()

if __name__ == "__main__":
    print()
    print("=" * 58)
    print("  AI Glasses -- Python Server v2.0 (Final)")
    print("  Gemini Flash + gTTS + WAV PCM Output")
    print("=" * 58)
    print()
    
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("[WARNING] Gemini API Key not set!")
        print("  1. Go to https://aistudio.google.com/apikey")
        print("  2. Create API Key (free)")
        print("  3. Set GEMINI_API_KEY in this file")
        print("  Or: set GEMINI_API_KEY=your_key_here")
        print()
    
    # Get local IP for display
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"
    
    print(f"  Configuration:")
    print(f"    Gemini Model  : {GEMINI_MODEL}")
    print(f"    Audio Format  : {'WAV PCM (16kHz/16bit/mono)' if FFMPEG_AVAILABLE else 'MP3 (ffmpeg not installed!)'}")
    print(f"    Debug Mode    : {'ON' if DEBUG_MODE else 'OFF'}")
    print(f"    ffmpeg        : {'Available' if FFMPEG_AVAILABLE else 'Not installed'}")
    print()
    print(f"  Server starting on:")
    print(f"    http://localhost:{SERVER_PORT}")
    print(f"    http://{local_ip}:{SERVER_PORT}")
    print()
    print(f"  ESP32 Config.h settings:")
    print(f'    #define PYTHON_SERVER_IP    "{local_ip}"')
    print(f"    #define PYTHON_SERVER_PORT  {SERVER_PORT}")
    print()
    print(f"  Endpoints:")
    print(f"    GET  /health   -- Check server status")
    print(f"    POST /analyze  -- Auto danger detection")
    print(f"    POST /describe -- Scene description (button press)")
    print(f"    GET  /beep     -- Download beep WAV")
    print()
    print(f"  Tips:")
    print(f"    - Watch this console for debug logs")
    print(f"    - Every request has a [REQ-XXXX] tracking ID")
    print(f"    - Latency breakdown shows timing per step")
    print()
    
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False)
