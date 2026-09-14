#include "AudioMod.h"
#include <math.h>

AudioMod Audio;

static const int kSampleRate = 16000;

AudioMod::AudioMod() {
    _isInitialized = false;
}

bool AudioMod::begin() {
    i2s_config_t i2s_config = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate          = (uint32_t)kSampleRate,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 1024,
        .use_apll             = false,
        .tx_desc_auto_clear   = true,
        .fixed_mclk           = 0
    };

    i2s_pin_config_t pin_config = {
        .mck_io_num   = I2S_PIN_NO_CHANGE,
        .bck_io_num   = I2S_SPK_BCLK,
        .ws_io_num    = I2S_SPK_LRC,
        .data_out_num = I2S_SPK_DOUT,
        .data_in_num  = I2S_PIN_NO_CHANGE
    };

    esp_err_t err = i2s_driver_install(I2S_PORT_SPEAKER, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[AUDIO] I2S driver install failed: %d\n", err);
        _isInitialized = false;
        return false;
    }

    err = i2s_set_pin(I2S_PORT_SPEAKER, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[AUDIO] I2S pin config failed: %d\n", err);
        _isInitialized = false;
        return false;
    }

    _isInitialized = true;
    Serial.println("[AUDIO] I2S speaker ready (open-ear / bone-conduction).");
    return true;
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

void AudioMod::_writeI2S(const int16_t* buf, int numSamples) {
    if (!_isInitialized) return;
    size_t bytesWritten;
    i2s_write(I2S_PORT_SPEAKER, buf, numSamples * sizeof(int16_t), &bytesWritten, portMAX_DELAY);
}

/*static*/ int AudioMod::_genSine(
    int frequency, int durationMs, float amplitude,
    int16_t* outBuf, int maxSamples, int sampleRate
) {
    int n = (sampleRate * durationMs) / 1000;
    if (n > maxSamples) n = maxSamples;
    int fade = n / 10;
    for (int i = 0; i < n; i++) {
        float env = 1.0f;
        if      (i < fade)      env = (float)i / fade;
        else if (i > n - fade)  env = (float)(n - i) / fade;
        float v = sinf(2.0f * M_PI * frequency * i / sampleRate) * amplitude * 32767.0f * env;
        outBuf[i] = (int16_t)constrain((int)v, -32768, 32767);
    }
    return n;
}

// ─────────────────────────────────────────────
// feedWAVData — skip 44-byte WAV header, push PCM to I2S
// ─────────────────────────────────────────────
void AudioMod::feedWAVData(const uint8_t* data, size_t length) {
    if (!_isInitialized || !data || length <= 44) return;
    // WAV header is always 44 bytes for our PCM format (16kHz/16bit/mono)
    const uint8_t* pcm = data + 44;
    size_t pcmLen = length - 44;
    feedTTSData(pcm, pcmLen);
}

void AudioMod::feedTTSData(const uint8_t* data, size_t length) {
    if (!_isInitialized || !data || length == 0) return;
    size_t bytesWritten;
    // Stream in 4 KB chunks to avoid blocking
    const size_t chunkSize = 4096;
    size_t offset = 0;
    while (offset < length) {
        size_t toWrite = min(chunkSize, length - offset);
        i2s_write(I2S_PORT_SPEAKER, data + offset, toWrite, &bytesWritten, portMAX_DELAY);
        offset += toWrite;
    }
}

// ─────────────────────────────────────────────
// Configurable tone
// ─────────────────────────────────────────────
void AudioMod::playTone(int frequency, int durationMs, float amplitude) {
    if (!_isInitialized) return;
    const int maxSamples = (kSampleRate * 3000) / 1000; // max 3 s
    int16_t* buf = (int16_t*)malloc(maxSamples * sizeof(int16_t));
    if (!buf) return;
    int n = _genSine(frequency, durationMs, amplitude, buf, maxSamples);
    _writeI2S(buf, n);
    free(buf);
}

// ─────────────────────────────────────────────
// HomeSight Earcons — PRD §4.2
// ─────────────────────────────────────────────

// P0 — Rising two-tone chime: 880 Hz + 1100 Hz
// Highest urgency — interrupts any pending audio
void AudioMod::playEarconP0() {
    if (!_isInitialized) {
        Serial.println("[AUDIO] Earcon P0 (no speaker)");
        return;
    }
    Serial.println("[AUDIO] ♦ P0 earcon");
    const int bufSz = 1024;
    int16_t buf[bufSz];
    int n;

    // First tone 880 Hz 150 ms
    n = _genSine(880, 150, 0.70f, buf, bufSz);
    _writeI2S(buf, n);
    // 50 ms silence
    memset(buf, 0, 100 * sizeof(int16_t));
    _writeI2S(buf, 100);
    // Second tone 1100 Hz 200 ms (louder)
    n = _genSine(1100, 200, 0.80f, buf, bufSz);
    _writeI2S(buf, n);
}

// P1 — Soft single pulse: 550 Hz 120 ms, low amplitude
void AudioMod::playEarconP1() {
    if (!_isInitialized) {
        Serial.println("[AUDIO] Earcon P1 (no speaker)");
        return;
    }
    Serial.println("[AUDIO] ◆ P1 earcon");
    playTone(550, 120, 0.40f);
}

// P2 — Directional triple blip: 660 Hz × 3, short gaps
void AudioMod::playEarconP2() {
    if (!_isInitialized) {
        Serial.println("[AUDIO] Earcon P2 (no speaker)");
        return;
    }
    Serial.println("[AUDIO] ● P2 earcon");
    const int bufSz = 512;
    int16_t buf[bufSz];
    int16_t silence[200];
    memset(silence, 0, sizeof(silence));

    for (int i = 0; i < 3; i++) {
        int n = _genSine(660, 80, 0.50f, buf, bufSz);
        _writeI2S(buf, n);
        if (i < 2) _writeI2S(silence, 200);  // 200 samples ≈ 12.5 ms gap
    }
}

// Legacy compatibility
void AudioMod::playAlertTone() {
    playEarconP0();
}

// ─────────────────────────────────────────────
// Proximity Click Tone — Object Locator (PRD §4.3)
// Frequency of clicks increases as user aligns with target.
// clickFreqHz: 2 (far) → 20 (close)
// ─────────────────────────────────────────────
void AudioMod::playProximityClicks(int clickFreqHz, int durationMs) {
    if (!_isInitialized) return;

    clickFreqHz = constrain(clickFreqHz, 1, 30);
    int totalSamples = (kSampleRate * durationMs) / 1000;
    int clickPeriod  = kSampleRate / clickFreqHz;       // samples per click
    int clickLen     = min(kSampleRate / 60, clickPeriod / 3); // 16 ms click

    const int bufSz = 256;
    int16_t buf[bufSz];

    int samplesPlayed = 0;
    while (samplesPlayed < totalSamples) {
        int posInPeriod = samplesPlayed % clickPeriod;

        int chunkSize = min(bufSz, totalSamples - samplesPlayed);
        for (int i = 0; i < chunkSize; i++) {
            int posI = (posInPeriod + i) % clickPeriod;
            int16_t v = 0;
            if (posI < clickLen) {
                float env = sinf(M_PI * posI / clickLen);
                float sine = sinf(2.0f * M_PI * 880.0f * posI / kSampleRate);
                v = (int16_t)(sine * 0.60f * 32767.0f * env);
            }
            buf[i] = v;
        }
        _writeI2S(buf, chunkSize);
        samplesPlayed += chunkSize;
    }
}

// ─────────────────────────────────────────────
// Confirmation chirp — ascending two-note
// ─────────────────────────────────────────────
void AudioMod::playConfirmChirp() {
    if (!_isInitialized) return;
    playTone(660, 80, 0.35f);
    playTone(880, 80, 0.35f);
}

// ─────────────────────────────────────────────
// System sound (named — reserved for SD card audio in future)
// ─────────────────────────────────────────────
void AudioMod::playSystemSound(const String& soundName) {
    Serial.println("[AUDIO] System sound: " + soundName);
    if (!_isInitialized) return;
    // Placeholder — SD card audio not in scope for Phase 1
    playConfirmChirp();
}
