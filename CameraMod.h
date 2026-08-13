#ifndef CAMERA_MOD_H
#define CAMERA_MOD_H

#include <Arduino.h>
#include "esp_camera.h"
#include "Config.h"

// Motion detection result
struct MotionResult {
    bool detected;
    float changePercent;   // % of bytes that changed significantly
    uint32_t changedPixels;
    uint32_t totalPixels;
};

class CameraMod {
public:
    CameraMod();
    bool begin();
    camera_fb_t* captureFrame();
    void releaseFrame(camera_fb_t* fb);
    void end();
    
    // Motion detection using JPEG data comparison (no framesize change!)
    MotionResult analyzeMotion(camera_fb_t* fb);

private:
    uint8_t* _prevSamples;    // Sampled bytes from previous JPEG
    size_t   _prevSampleCount;
    bool     _hasPrevFrame;
    static const int SAMPLE_COUNT = 512; // Number of sample points from JPEG
};

extern CameraMod Camera;

#endif
