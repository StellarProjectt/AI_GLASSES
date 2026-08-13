#ifndef CLOUD_API_H
#define CLOUD_API_H

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

// Response from Python server
struct ServerResponse {
    bool success;
    bool hasAlert;
    String message;
    uint8_t* audioData;   // MP3 audio data (dynamically allocated, caller must free)
    size_t audioLength;
    float elapsed;
};

class CloudAPI {
public:
    CloudAPI();
    
    // WiFi - Station Mode
    bool connectWiFi();
    bool isConnected();
    
    // WiFi - AP Mode (fallback)
    bool startAP();
    String getIP();
    
    // Python Server communication
    bool checkServer();  // ตรวจว่า Python server พร้อมไหม
    
    // ส่งภาพไปตรวจอันตราย
    ServerResponse analyzeImage(camera_fb_t* fb);
    
    // ส่งภาพไปอธิบาย (กดปุ่ม)
    ServerResponse describeImage(camera_fb_t* fb);
    
    // Offline alert (ถ้าไม่มี server)
    void speakAlert(const String& message);

private:
    bool _apMode;
    String _serverUrl;
    
    // ส่ง JPEG ไป endpoint ที่กำหนด
    ServerResponse sendImage(camera_fb_t* fb, const String& endpoint);
};

extern CloudAPI Cloud;

#endif
