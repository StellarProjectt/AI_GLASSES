#include "AudioMod.h"
#include <math.h>

AudioMod Audio;

AudioMod::AudioMod() {
    _isInitialized = false;
}

bool AudioMod::begin() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .mck_io_num = I2S_PIN_NO_CHANGE,
        .bck_io_num = I2S_SPK_BCLK,
        .ws_io_num = I2S_SPK_LRC,
        .data_out_num = I2S_SPK_DOUT,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    esp_err_t err = i2s_driver_install(I2S_PORT_SPEAKER, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("Failed installing I2S driver for speaker: %d\n", err);
        // Don't return false — system can work without speaker
        _isInitialized = false;
        return false;
    }

    err = i2s_set_pin(I2S_PORT_SPEAKER, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("Failed setting I2S pins for speaker: %d\n", err);
        _isInitialized = false;
        return false;
    }

    _isInitialized = true;
    return true;
}

void AudioMod::playSystemSound(const String& soundName) {
    Serial.println("[AUDIO] " + soundName);
    if (!_isInitialized) return;
    // Placeholder — would play pre-recorded WAV from SD in production
}

// Generate and play a sine wave tone via I2S
void AudioMod::playTone(int frequency, int durationMs) {
    if (!_isInitialized) return;
    
    const int sampleRate = 16000;
    int totalSamples = (sampleRate * durationMs) / 1000;
    const int bufSize = 256;
    int16_t buf[bufSize];
    
    int samplesWritten = 0;
    while (samplesWritten < totalSamples) {
        int chunkSize = min(bufSize, totalSamples - samplesWritten);
        for (int i = 0; i < chunkSize; i++) {
            float t = (float)(samplesWritten + i) / (float)sampleRate;
            // Sine wave with volume envelope
            float envelope = 1.0f;
            if (samplesWritten + i < sampleRate / 20) {
                envelope = (float)(samplesWritten + i) / (sampleRate / 20); // fade in
            }
            if (samplesWritten + i > totalSamples - sampleRate / 20) {
                envelope = (float)(totalSamples - samplesWritten - i) / (sampleRate / 20); // fade out
            }
            buf[i] = (int16_t)(sinf(2.0f * M_PI * frequency * t) * 16000 * envelope);
        }
        size_t bytesWritten;
        i2s_write(I2S_PORT_SPEAKER, buf, chunkSize * sizeof(int16_t), &bytesWritten, portMAX_DELAY);
        samplesWritten += chunkSize;
    }
}

// Play a warning alert: two quick beeps
void AudioMod::playAlertTone() {
    if (!_isInitialized) {
        Serial.println("[AUDIO] Alert tone (no speaker)");
        return;
    }
    playTone(880, 150);   // High A note, 150ms
    delay(50);
    playTone(1100, 200);  // Higher pitch, 200ms
}

void AudioMod::feedTTSData(const uint8_t* data, size_t length) {
    if (!_isInitialized) return;
    size_t bytesWritten;
    i2s_write(I2S_PORT_SPEAKER, data, length, &bytesWritten, portMAX_DELAY);
}
