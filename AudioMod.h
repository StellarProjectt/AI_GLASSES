#ifndef AUDIO_MOD_H
#define AUDIO_MOD_H

#include <Arduino.h>
#include "driver/i2s.h"
#include "Config.h"

// I2S Port for Speaker
#define I2S_PORT_SPEAKER I2S_NUM_0

class AudioMod {
public:
    AudioMod();
    bool begin();

    // Feed raw WAV PCM data (from server audio_base64 response)
    // Skips the 44-byte WAV header and sends raw PCM to I2S.
    void feedWAVData(const uint8_t* data, size_t length);

    // Feed raw PCM samples directly (no header)
    void feedTTSData(const uint8_t* data, size_t length);

    // Earcon tones — generated on-device as fallback when server audio unavailable
    void playEarconP0();    // P0 Head hazard: rising two-tone chime
    void playEarconP1();    // P1 Floor obstacle: soft single pulse
    void playEarconP2();    // P2 Dynamic object: directional triple blip

    // Legacy alert tone (maps to playEarconP0 for compatibility)
    void playAlertTone();

    // Configurable tone generator
    void playTone(int frequency, int durationMs, float amplitude = 0.5f);

    // Proximity click for Object Locator mode
    // clickFreqHz: 2 Hz (far) → 20 Hz (close), durationMs total
    void playProximityClicks(int clickFreqHz, int durationMs = 1500);

    // Confirmation chirp (short ascending pair — used at boot and mode change)
    void playConfirmChirp();

    // System sound (named — for future SD card audio)
    void playSystemSound(const String& soundName);

    bool isInitialized() const { return _isInitialized; }

private:
    bool _isInitialized;

    // Write a block of int16_t samples to I2S
    void _writeI2S(const int16_t* buf, int numSamples);

    // Generate sine PCM into a heap buffer (caller must free)
    // Returns allocated byte count.
    static int _genSine(
        int frequency, int durationMs, float amplitude,
        int16_t* outBuf, int maxSamples,
        int sampleRate = 16000
    );
};

extern AudioMod Audio;

#endif // AUDIO_MOD_H
