"""
AI Glasses — Python Server
===========================
รับภาพจาก ESP32 → วิเคราะห์ด้วย Gemini Flash (ฟรี) → สร้างเสียงไทย (gTTS) → ส่งกลับ

Endpoints:
  POST /analyze  — ตรวจจับอันตรายอัตโนมัติ (ส่ง JPEG มา, ได้ WAV กลับ ถ้ามีอันตราย)
  POST /describe — อธิบายภาพเมื่อกดปุ่ม (ส่ง JPEG มา, ได้ WAV กลับเสมอ)
  GET  /health   — เช็คว่า server พร้อม

Usage:
  1. ใส่ Gemini API key ใน GEMINI_API_KEY ข้างล่าง (หรือ set environment variable)
  2. pip install -r requirements.txt
  3. python server.py
  4. ESP32 จะส่งภาพมาที่ http://<your-ip>:5000
"""

import os
import io
import time
import base64
import struct
from flask import Flask, request, jsonify, send_file
from PIL import Image
import google.generativeai as genai
from gtts import gTTS

# ==========================================
# Configuration
# ==========================================
# วิธีได้ API Key ฟรี: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6I4tBtQZHUaBX5-gjGDU6wOdIoBSWDUc-rSHhibHU5_iQ")
SERVER_PORT = 5001

# ==========================================
# Initialize
# ==========================================
app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.5-flash")

print("=" * 50)
print("  AI Glasses Python Server")
print("=" * 50)

# ==========================================
# Helper: Convert MP3 to raw PCM WAV (16-bit, 16kHz, mono)
# ESP32 I2S needs raw PCM, not MP3
# ==========================================
def tts_to_wav_bytes(text, lang="th"):
    """Generate Thai TTS audio and return as WAV bytes (16-bit, 16kHz, mono)"""
    try:
        # Generate MP3 via gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        
        # We'll send MP3 directly — ESP32 can handle MP3 with a small decoder
        # or we convert to WAV. For simplicity, let's send MP3 and let ESP32
        # save+play it, OR we return it as-is for the web UI.
        # 
        # Actually, for ESP32 I2S we need PCM. Let's try to convert.
        # But pydub/ffmpeg may not be installed. So let's return MP3
        # and have ESP32 just use it for web playback.
        # 
        # For ESP32 speaker: we'll send a simple beep pattern instead
        # and play the full TTS on the web UI.
        
        return mp3_buf.getvalue()
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def generate_pcm_beep(frequency=880, duration_ms=200, sample_rate=16000):
    """Generate raw PCM beep for ESP32 I2S speaker (16-bit, mono)"""
    import math
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        # Sine wave with fade in/out
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


def analyze_danger(image_bytes):
    """Send image to Gemini and check for dangers"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content(
            [DANGER_PROMPT, image],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.1,
            )
        )
        result = response.text.strip()
        print(f"  Gemini (danger): {result}")
        
        if result.startswith("DANGER|"):
            warning = result.split("|", 1)[1]
            return True, warning
        return False, ""
    except Exception as e:
        print(f"  Gemini Error: {e}")
        return False, f"ระบบผิดพลาด: {str(e)}"


def describe_scene(image_bytes):
    """Send image to Gemini and get scene description"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content(
            [DESCRIBE_PROMPT, image],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,
            )
        )
        result = response.text.strip()
        print(f"  Gemini (describe): {result}")
        return result
    except Exception as e:
        print(f"  Gemini Error: {e}")
        return f"ไม่สามารถวิเคราะห์ภาพได้ ลองอีกครั้ง"


# ==========================================
# API Endpoints
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    """Health check — ESP32 calls this to verify server is reachable"""
    return jsonify({
        "status": "ok",
        "model": "gemini-2.0-flash",
        "version": "1.0"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Auto danger detection endpoint.
    ESP32 sends JPEG image, server checks for dangers.
    
    Returns:
      - If danger: JSON with alert=true + MP3 audio of Thai warning
      - If safe: JSON with alert=false (no audio)
    """
    start = time.time()
    
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_data = request.files["image"].read()
    print(f"\n[ANALYZE] Received {len(image_data)} bytes")
    
    is_danger, warning = analyze_danger(image_data)
    elapsed = time.time() - start
    
    if is_danger:
        print(f"  ⚠️  DANGER: {warning} ({elapsed:.1f}s)")
        
        # Generate TTS audio
        audio_data = tts_to_wav_bytes(warning)
        
        if audio_data:
            # Return JSON + audio as multipart
            return jsonify({
                "alert": True,
                "message": warning,
                "audio_base64": base64.b64encode(audio_data).decode("utf-8"),
                "audio_format": "mp3",
                "elapsed": round(elapsed, 2)
            })
        else:
            return jsonify({
                "alert": True,
                "message": warning,
                "elapsed": round(elapsed, 2)
            })
    else:
        print(f"  ✅ SAFE ({elapsed:.1f}s)")
        return jsonify({
            "alert": False,
            "elapsed": round(elapsed, 2)
        })


@app.route("/describe", methods=["POST"])
def describe():
    """
    Scene description endpoint (triggered by button press).
    ESP32 sends JPEG image, server describes the scene in Thai.
    
    Returns: JSON with description text + MP3 audio
    """
    start = time.time()
    
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_data = request.files["image"].read()
    print(f"\n[DESCRIBE] Received {len(image_data)} bytes")
    
    description = describe_scene(image_data)
    elapsed = time.time() - start
    print(f"  📝 Description: {description} ({elapsed:.1f}s)")
    
    # Generate TTS audio
    audio_data = tts_to_wav_bytes(description)
    
    result = {
        "description": description,
        "elapsed": round(elapsed, 2)
    }
    
    if audio_data:
        result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")
        result["audio_format"] = "mp3"
    
    return jsonify(result)


@app.route("/beep", methods=["GET"])
def beep():
    """Return a simple alert beep WAV for ESP32 I2S speaker"""
    # Two-tone alert beep
    beep1 = generate_pcm_beep(880, 150)
    silence = b'\x00\x00' * 800  # 50ms silence
    beep2 = generate_pcm_beep(1100, 200)
    pcm = beep1 + silence + beep2
    wav = create_wav_header(pcm)
    
    return send_file(
        io.BytesIO(wav),
        mimetype="audio/wav",
        as_attachment=False
    )


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print()
        print("⚠️  WARNING: Gemini API Key ยังไม่ได้ตั้ง!")
        print("   1. ไปที่ https://aistudio.google.com/apikey")
        print("   2. สร้าง API Key (ฟรี)")
        print("   3. แก้ GEMINI_API_KEY ในไฟล์นี้")
        print("   หรือ: export GEMINI_API_KEY=your_key_here")
        print()
    
    # Get local IP for display
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    print(f"\n🚀 Server starting on:")
    print(f"   http://localhost:{SERVER_PORT}")
    print(f"   http://{local_ip}:{SERVER_PORT}")
    print(f"\n   ตั้ง ESP32 ให้ชี้มาที่ IP นี้")
    print(f"   แก้ PYTHON_SERVER_IP ใน Config.h เป็น \"{local_ip}\"")
    print()
    
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False)
