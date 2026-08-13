#include "CameraMod.h"

CameraMod Camera;

CameraMod::CameraMod() {
    _prevSamples = nullptr;
    _prevSampleCount = 0;
    _hasPrevFrame = false;
}

bool CameraMod::begin() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    
    // OV3660: use 10MHz xclk for stability (20MHz can be unstable)
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode    = CAMERA_GRAB_LATEST;  // Always get the latest frame
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count     = 1;

    // Set resolution based on available PSRAM
    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;   // 640x480
        config.jpeg_quality = 10;
        config.fb_count     = 2;  // 2 buffers for smooth streaming
        Serial.println("PSRAM found — using VGA, 2 framebuffers");
    } else {
        config.frame_size   = FRAMESIZE_QVGA;  // 320x240
        config.jpeg_quality = 14;
        config.fb_count     = 1;
        Serial.println("No PSRAM — using QVGA, 1 framebuffer");
    }

    // Initialize the camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    // OV3660 specific settings
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        if (s->id.PID == OV3660_PID) {
            Serial.println("OV3660 detected! Applying optimal settings...");
            s->set_vflip(s, 1);        // OV3660 is often mounted upside down
            s->set_brightness(s, 1);    // Slightly brighter
            s->set_saturation(s, -1);   // Less saturated for clearer image
        } else {
            Serial.printf("Camera PID: 0x%x\n", s->id.PID);
        }
    }

    // Allocate sample buffer for motion detection
    _prevSamples = (uint8_t*)malloc(SAMPLE_COUNT);
    if (!_prevSamples) {
        Serial.println("Warning: Could not allocate motion buffer");
    }

    // Warm-up: discard the first few frames (OV3660 needs this)
    Serial.print("Camera warm-up");
    for (int i = 0; i < 5; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
            Serial.printf(" [%u bytes]", fb->len);
            esp_camera_fb_return(fb);
        } else {
            Serial.print(" [fail]");
        }
        delay(100);
    }
    Serial.println(" done!");

    Serial.println("Camera initialized successfully!");
    return true;
}

camera_fb_t* CameraMod::captureFrame() {
    return esp_camera_fb_get();
}

void CameraMod::releaseFrame(camera_fb_t* fb) {
    if (fb) {
        esp_camera_fb_return(fb);
    }
}

// Analyze motion by comparing sampled JPEG bytes between frames.
// This is called WITH an already-captured frame — no extra capture needed!
// JPEG data naturally encodes brightness/texture, so comparing sampled bytes
// gives a good approximation of scene change without any resolution switching.
MotionResult CameraMod::analyzeMotion(camera_fb_t* fb) {
    MotionResult result = {false, 0.0f, 0, 0};
    
    if (!_prevSamples || !fb || fb->len < 100) {
        return result;
    }

    // Sample evenly-spaced bytes from the JPEG data
    // Skip the JPEG header (first ~50 bytes) and footer
    size_t dataStart = 50;  // Skip JPEG header markers
    size_t dataEnd = fb->len > 50 ? fb->len - 2 : fb->len;  // Skip JPEG EOF marker
    size_t dataLen = dataEnd - dataStart;
    
    if (dataLen < SAMPLE_COUNT) {
        return result;
    }

    float step = (float)dataLen / (float)SAMPLE_COUNT;
    
    uint8_t currentSamples[SAMPLE_COUNT];
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        size_t idx = dataStart + (size_t)(i * step);
        if (idx < fb->len) {
            currentSamples[i] = fb->buf[idx];
        } else {
            currentSamples[i] = 0;
        }
    }

    result.totalPixels = SAMPLE_COUNT;

    if (_hasPrevFrame) {
        // Compare samples
        uint32_t changed = 0;
        for (int i = 0; i < SAMPLE_COUNT; i++) {
            int diff = abs((int)currentSamples[i] - (int)_prevSamples[i]);
            if (diff > MOTION_THRESHOLD) {
                changed++;
            }
        }
        
        result.changedPixels = changed;
        result.changePercent = (float)changed / (float)SAMPLE_COUNT * 100.0f;
        result.detected = (result.changePercent > MOTION_PERCENT_ALERT);
    }

    // Store current as previous
    memcpy(_prevSamples, currentSamples, SAMPLE_COUNT);
    _hasPrevFrame = true;
    _prevSampleCount = SAMPLE_COUNT;
    
    return result;
}

void CameraMod::end() {
    if (_prevSamples) {
        free(_prevSamples);
        _prevSamples = nullptr;
    }
    esp_camera_deinit();
}
