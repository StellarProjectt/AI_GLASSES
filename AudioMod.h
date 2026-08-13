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
    
    // Play system pre-recorded sounds (dummy implementation for now)
    void playSystemSound(const String& soundName);
    
    // Play an alert buzzer tone via I2S
    void playAlertTone();
    
    // Play a specific frequency tone
    void playTone(int frequency, int durationMs);
    
    // Feed raw PCM data from cloud TTS
    void feedTTSData(const uint8_t* data, size_t length);

private:
    bool _isInitialized;
};

extern AudioMod Audio;

#endif
