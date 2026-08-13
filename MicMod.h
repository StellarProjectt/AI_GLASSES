#ifndef MIC_MOD_H
#define MIC_MOD_H

#include <Arduino.h>
#include "driver/i2s.h"
#include "Config.h"

// I2S Port for Microphone
#define I2S_PORT_MIC I2S_NUM_1

class MicMod {
public:
    MicMod();
    bool begin();
    
    // Read audio data from I2S mic
    size_t readAudio(uint8_t* buffer, size_t bufferSize);

private:
    bool _isInitialized;
};

extern MicMod Mic;

#endif
