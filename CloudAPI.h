#ifndef CLOUD_API_H
#define CLOUD_API_H

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "Config.h"

// ─────────────────────────────────────────────
// Server response — parsed from HomeSight Python server v3.0
// ─────────────────────────────────────────────
struct ServerResponse {
    bool     success;
    bool     systemReady;     // false → server has no model loaded / not ready
                               // (distinct from "path clear" — must not stay silent)
    bool     hasAlert;        // true → hazard detected (play audio)
    String   message;         // Prompt string: "Watch head — cabinet"
    String   earconCode;      // "HIGH" | "LOW" | "DIR" | ""
    int      priority;        // HazardPriority enum value (0-3)
    String   side;            // "left" | "center" | "right"
    float    distanceNorm;    // 0.0 (close) – 1.0 (far), for proximity clicks
    uint8_t* audioData;       // WAV PCM bytes (heap-allocated — caller must free)
    size_t   audioLength;     // bytes
    uint8_t* clickAudioData;  // Proximity click WAV (Object Locator only)
    size_t   clickAudioLength;
    float    elapsed;         // Server-side processing time (seconds)
};

// ─────────────────────────────────────────────
// CloudAPI — LAN communication with HomeSight Python server
// No internet connection required.
// ─────────────────────────────────────────────
class CloudAPI {
public:
    CloudAPI();

    // WiFi — Station Mode
    bool   connectWiFi();
    bool   isConnected();

    // WiFi — AP Mode (fallback when home WiFi unavailable)
    bool   startAP();
    String getIP();

    // Server health check — call at boot and every 10 s if disconnected
    bool checkServer();

    // /analyze — continuous hazard detection (every AUTO_DETECT_INTERVAL_MS)
    ServerResponse analyzeImage(camera_fb_t* fb);

    // /describe — scene description triggered by single button press
    ServerResponse describeImage(camera_fb_t* fb);

    // /search  — object locator triggered by double-tap + voice query
    ServerResponse searchObject(camera_fb_t* fb, const String& query);

    // Fallback alert tone when server is unreachable
    void speakAlert(const String& message);

private:
    bool   _apMode;
    String _serverUrl;

    // Core: multipart-POST a JPEG frame to a server endpoint
    ServerResponse sendImage(camera_fb_t* fb, const String& endpoint,
                             const String& extraField = "",
                             const String& extraValue = "");

    // Parse base64 audio field from JSON doc into heap buffer
    static uint8_t* _decodeAudioField(const String& b64, size_t& outLen);
};

extern CloudAPI Cloud;

#endif // CLOUD_API_H
