# -*- coding: utf-8 -*-
"""
HomeSight — Audio Cache
========================
Pre-renders all HomeSight alert phrases to WAV PCM at server startup.
Provides O(1) dictionary lookup for zero-latency audio retrieval.

WAV spec: 16 kHz, 16-bit, mono (native ESP32 I2S format).

Falls back to pyttsx3 real-time synthesis for any phrase not in the cache.
"""

from __future__ import annotations
import io
import struct
import math
import logging
import threading
from typing import Optional

logger = logging.getLogger("AI_Glasses.AudioCache")

# ─────────────────────────────────────────────
# All HomeSight Alert Prompts (PRD §4.2)
# ─────────────────────────────────────────────

# These are all phrases the classifier can ever produce.
# Pre-rendering them at startup eliminates TTS latency from the hot path.
ALERT_PROMPTS: list[str] = [
    # P0 — Head hazards
    "Watch head — cabinet",
    "Watch head — door",
    "Watch head — fridge",
    "Watch head — microwave",
    "Watch head — screen",
    "Watch head — lamp",
    "Watch head — shelf",
    "Watch head — fixture",
    "Watch head — laptop",
    # P1 — Floor obstacles
    "Watch steps — fan",
    "Watch steps — chair",
    "Watch steps — table",
    "Watch steps — box",
    "Watch steps — bag",
    "Watch steps — bottle",
    "Watch steps — plant",
    "Watch steps — suitcase",
    "Watch steps — bench",
    "Watch steps — cord",
    "Watch steps — cable",
    "Watch steps — bin",
    "Watch steps — basket",
    "Watch steps — ball",
    "Watch left — chair",
    "Watch right — chair",
    "Watch left — fan",
    "Watch right — fan",
    "Watch left — table",
    "Watch right — table",
    "Watch left — box",
    "Watch right — box",
    "Watch left — bag",
    "Watch right — bag",
    # P2 — Dynamic
    "Pet ahead",
    "Person passing",
    # System
    "Battery low",
    "Connected",
    "Ready",
]

# ─────────────────────────────────────────────
# WAV / PCM Utilities
# ─────────────────────────────────────────────

SAMPLE_RATE = 16000
BITS        = 16
CHANNELS    = 1


def _make_wav_header(pcm_len: int) -> bytes:
    """Create a minimal RIFF WAV header for raw PCM data."""
    return struct.pack(
        "<4sI4s"    # RIFF chunk
        "4sIHHIIHH" # fmt  chunk
        "4sI",      # data chunk
        b"RIFF", 36 + pcm_len, b"WAVE",
        b"fmt ", 16, 1, CHANNELS, SAMPLE_RATE,
        SAMPLE_RATE * CHANNELS * BITS // 8,
        CHANNELS * BITS // 8, BITS,
        b"data", pcm_len,
    )


def _wrap_wav(pcm_data: bytes) -> bytes:
    return _make_wav_header(len(pcm_data)) + pcm_data


def generate_earcon(
    priority: str,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """
    Generate a synthetic earcon WAV for each priority class.

    P0 HIGH  → two-tone rising chime (880 Hz + 1100 Hz), 200 ms
    P1 LOW   → single soft pulse (550 Hz), 120 ms
    P2 DIR   → triple blip (660 Hz × 3), 80 ms each
    """
    def sine_pcm(freq: float, duration_ms: int, amplitude: float = 0.6) -> bytes:
        n = int(sample_rate * duration_ms / 1000)
        fade = n // 8
        buf = bytearray(n * 2)
        for i in range(n):
            env = 1.0
            if i < fade:
                env = i / fade
            elif i > n - fade:
                env = (n - i) / fade
            v = int(math.sin(2 * math.pi * freq * i / sample_rate) * amplitude * 32767 * env)
            v = max(-32768, min(32767, v))
            struct.pack_into("<h", buf, i * 2, v)
        return bytes(buf)

    silence_50ms = b"\x00\x00" * (sample_rate * 50 // 1000)

    if priority == "HIGH":
        pcm = sine_pcm(880, 150) + silence_50ms + sine_pcm(1100, 200, 0.7)
    elif priority == "LOW":
        pcm = sine_pcm(550, 120, 0.4)
    elif priority == "DIR":
        blip = sine_pcm(660, 80, 0.5)
        pcm = blip + silence_50ms + blip + silence_50ms + blip
    else:
        pcm = sine_pcm(440, 80, 0.3)

    return _wrap_wav(pcm)


# ─────────────────────────────────────────────
# pyttsx3 synthesis (offline, no network)
# ─────────────────────────────────────────────

def _synth_with_macos_say(text: str) -> Optional[bytes]:
    """
    Use macOS built-in 'say' command to synthesise AIFF, then convert to WAV.
    Available on every Mac — no pip install required.
    Voice: Samantha (English) at 200 wpm.
    """
    import tempfile, os
    try:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            aiff_path = tmp.name
        wav_path = aiff_path.replace(".aiff", ".wav")

        # Generate AIFF via macOS say
        result = subprocess.run(
            ["say", "-v", "Samantha", "-r", "200", "-o", aiff_path, text],
            capture_output=True, timeout=10
        )
        if result.returncode != 0 or not os.path.exists(aiff_path):
            return None

        # Convert AIFF → WAV PCM 16kHz/16bit/mono via ffmpeg
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path,
             "-f", "wav", "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True, timeout=15
        )
        if proc.returncode == 0 and os.path.exists(wav_path):
            with open(wav_path, "rb") as f:
                data = f.read()
            os.unlink(aiff_path)
            os.unlink(wav_path)
            return data

        # ffmpeg not available — return raw AIFF and let ESP32 skip it
        os.unlink(aiff_path)
        return None

    except FileNotFoundError:
        # 'say' not available (non-Mac)
        return None
    except Exception as e:
        logger.debug(f"macOS say failed: {e}")
        return None


def _synth_with_pyttsx3(text: str) -> Optional[bytes]:
    """
    Synthesise speech offline using pyttsx3.
    Returns WAV PCM bytes (16 kHz / 16-bit / mono) or None on failure.
    """
    try:
        import pyttsx3
        import tempfile, os

        engine = pyttsx3.init()
        engine.setProperty("rate", 190)
        engine.setProperty("volume", 0.95)

        voices = engine.getProperty("voices")
        for v in voices:
            if "english" in v.name.lower() or "en" in (v.id or "").lower():
                engine.setProperty("voice", v.id)
                break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        return _resample_wav_if_needed(data)

    except Exception as e:
        logger.debug(f"pyttsx3 synthesis failed: {e}")
        return None


def _synth_with_gtts(text: str) -> Optional[bytes]:
    """Fallback: gTTS (requires internet). Used only if all offline methods fail."""
    try:
        from gtts import gTTS
        mp3_buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(mp3_buf)
        mp3_bytes = mp3_buf.getvalue()
        # Convert MP3 → WAV via ffmpeg if available
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav",
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16", "-y", "pipe:1"],
            input=mp3_bytes, capture_output=True, timeout=20
        )
        if proc.returncode == 0:
            return proc.stdout
        return mp3_bytes
    except Exception as e:
        logger.debug(f"gTTS fallback failed: {e}")
        return None


def _synth_speech(text: str) -> Optional[bytes]:
    """
    Try TTS engines in order of preference:
      1. macOS say  (always available on Mac, no pip needed)
      2. pyttsx3    (offline, cross-platform)
      3. gTTS       (requires internet, last resort)
    """
    wav = _synth_with_macos_say(text)
    if wav:
        return wav
    wav = _synth_with_pyttsx3(text)
    if wav:
        return wav
    logger.warning(f"All offline TTS failed for '{text}' — trying gTTS (needs internet)")
    return _synth_with_gtts(text)


def _resample_wav_if_needed(wav_bytes: bytes) -> Optional[bytes]:
    """
    If the WAV sample rate != 16000 Hz, convert it using ffmpeg.
    If ffmpeg isn't available, return as-is and let ESP32 handle it.
    """
    import subprocess, struct as st

    # Read sample rate from WAV header (bytes 24-27)
    if len(wav_bytes) < 44:
        return wav_bytes
    sr = st.unpack_from("<I", wav_bytes, 24)[0]
    if sr == SAMPLE_RATE:
        return wav_bytes

    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0",
             "-f", "wav", "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-sample_fmt", "s16", "-y", "pipe:1"],
            input=wav_bytes,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout
    except Exception:
        pass
    return wav_bytes


# ─────────────────────────────────────────────
# Audio Cache Class
# ─────────────────────────────────────────────

class AudioCache:
    """
    Thread-safe pre-rendered audio cache for HomeSight alerts.
    Call `warmup()` once at server startup (runs in background thread).
    """

    def __init__(self):
        self._cache: dict[str, bytes] = {}
        self._earcons: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._ready = False

    def warmup(self, blocking: bool = False) -> None:
        """
        Pre-render all ALERT_PROMPTS to WAV.
        If blocking=False (default), runs in a background thread so the
        server starts immediately and warms up concurrently.
        """
        if blocking:
            self._render_all()
        else:
            t = threading.Thread(target=self._render_all, daemon=True)
            t.start()

    def _render_all(self) -> None:
        logger.info("[AudioCache] Warming up — pre-rendering alert phrases...")

        # Earcons (instant, pure synthesis — no TTS)
        for code in ("HIGH", "LOW", "DIR", "NONE"):
            wav = generate_earcon(code)
            with self._lock:
                self._earcons[code] = wav
        logger.info("[AudioCache] Earcons ready.")

        # Speech prompts
        ok = 0
        for phrase in ALERT_PROMPTS:
            wav = _synth_speech(phrase)
            if wav:
                with self._lock:
                    self._cache[phrase] = wav
                ok += 1
            else:
                logger.warning(f"[AudioCache] Failed to pre-render: '{phrase}'")

        with self._lock:
            self._ready = True

        logger.info(f"[AudioCache] Ready — {ok}/{len(ALERT_PROMPTS)} phrases cached.")

    def get_speech(self, phrase: str) -> Optional[bytes]:
        """
        Return pre-rendered WAV bytes for the given phrase.
        Synthesises on-the-fly if not in cache (slower path).
        """
        with self._lock:
            cached = self._cache.get(phrase)
        if cached:
            return cached

        # Miss — synthesise and cache for next time
        logger.warning(f"[AudioCache] Cache miss, synthesising: '{phrase}'")
        wav = _synth_speech(phrase)
        if wav:
            with self._lock:
                self._cache[phrase] = wav
        return wav

    def get_earcon(self, code: str) -> bytes:
        """Return earcon WAV bytes for a given priority code."""
        with self._lock:
            return self._earcons.get(code, self._earcons.get("NONE", b""))

    def get_combined(self, earcon_code: str, phrase: str) -> bytes:
        """
        Returns earcon + speech concatenated as a single WAV.
        Both share the same spec (16kHz/16bit/mono), so we just
        concatenate the PCM data sections and rebuild the header.
        """
        earcon_wav = self.get_earcon(earcon_code)
        speech_wav = self.get_speech(phrase)

        if not speech_wav:
            return earcon_wav or b""
        if not earcon_wav:
            return speech_wav

        # Extract raw PCM from each WAV (skip 44-byte header)
        earcon_pcm = earcon_wav[44:]
        speech_pcm = speech_wav[44:]

        # 100 ms silence between earcon and speech
        silence = b"\x00\x00" * (SAMPLE_RATE * 100 // 1000)

        combined_pcm = earcon_pcm + silence + speech_pcm
        return _wrap_wav(combined_pcm)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready


# Module-level singleton — imported by server.py
audio_cache = AudioCache()
