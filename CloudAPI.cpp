#include "CloudAPI.h"
#include "Config.h"
#include "AudioMod.h"
#include <ArduinoJson.h>
#include <mbedtls/base64.h>

CloudAPI Cloud;

CloudAPI::CloudAPI() {
    _apMode = false;
    _serverUrl = "http://" + String(PYTHON_SERVER_IP) + ":" + String(PYTHON_SERVER_PORT);
}

// ==========================================
// WiFi
// ==========================================
bool CloudAPI::connectWiFi() {
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 30) {
        delay(500);
        Serial.print(".");
        retries++;
    }
    Serial.println();
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("WiFi Connected!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
        return true;
    }
    Serial.println("WiFi Connection Failed.");
    return false;
}

bool CloudAPI::startAP() {
    Serial.println("Starting WiFi Access Point...");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASS);
    delay(500);
    _apMode = true;
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());
    return true;
}

String CloudAPI::getIP() {
    if (_apMode) return WiFi.softAPIP().toString();
    return WiFi.localIP().toString();
}

bool CloudAPI::isConnected() {
    if (_apMode) return true;
    return WiFi.status() == WL_CONNECTED;
}

// ==========================================
// Python Server Communication
// ==========================================
bool CloudAPI::checkServer() {
    if (!isConnected()) return false;
    
    HTTPClient http;
    http.setTimeout(3000);
    http.begin(_serverUrl + "/health");
    int code = http.GET();
    http.end();
    
    return code == 200;
}

// Core function: send JPEG to Python server endpoint
ServerResponse CloudAPI::sendImage(camera_fb_t* fb, const String& endpoint) {
    ServerResponse resp = {false, false, "", nullptr, 0, 0};
    
    if (!isConnected() || !fb) return resp;
    
    HTTPClient http;
    http.setTimeout(15000); // 15 second timeout (Gemini can be slow)
    
    String url = _serverUrl + endpoint;
    http.begin(url);
    
    // Build multipart form data
    String boundary = "----ESP32Boundary";
    http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
    
    // Build the multipart body
    String bodyStart = "--" + boundary + "\r\n";
    bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n";
    bodyStart += "Content-Type: image/jpeg\r\n\r\n";
    
    String bodyEnd = "\r\n--" + boundary + "--\r\n";
    
    size_t totalLen = bodyStart.length() + fb->len + bodyEnd.length();
    
    // Allocate buffer for the full request body
    uint8_t* body = (uint8_t*)ps_malloc(totalLen);
    if (!body) {
        body = (uint8_t*)malloc(totalLen);
    }
    if (!body) {
        Serial.println("[API] Failed to allocate request buffer");
        http.end();
        return resp;
    }
    
    // Assemble the body
    size_t offset = 0;
    memcpy(body + offset, bodyStart.c_str(), bodyStart.length());
    offset += bodyStart.length();
    memcpy(body + offset, fb->buf, fb->len);
    offset += fb->len;
    memcpy(body + offset, bodyEnd.c_str(), bodyEnd.length());
    
    // Send POST request
    int httpCode = http.POST(body, totalLen);
    free(body);
    
    if (httpCode != 200) {
        Serial.printf("[API] HTTP Error: %d\n", httpCode);
        http.end();
        return resp;
    }
    
    // Parse JSON response
    String payload = http.getString();
    http.end();
    
    DynamicJsonDocument doc(8192);
    DeserializationError err = deserializeJson(doc, payload);
    if (err) {
        Serial.printf("[API] JSON Parse Error: %s\n", err.c_str());
        return resp;
    }
    
    resp.success = true;
    resp.elapsed = doc["elapsed"] | 0.0f;
    
    // Check for alert (danger detection)
    if (doc.containsKey("alert")) {
        resp.hasAlert = doc["alert"].as<bool>();
        if (resp.hasAlert) {
            resp.message = doc["message"].as<String>();
        }
    }
    
    // Check for description
    if (doc.containsKey("description")) {
        resp.message = doc["description"].as<String>();
    }
    
    // Check for audio data (base64 encoded MP3)
    if (doc.containsKey("audio_base64")) {
        String audioB64 = doc["audio_base64"].as<String>();
        if (audioB64.length() > 0) {
            // Decode base64 to binary using mbedtls
            size_t expected_len = 0;
            
            mbedtls_base64_decode(nullptr, 0, &expected_len, 
                (const unsigned char*)audioB64.c_str(), audioB64.length());
                
            resp.audioLength = expected_len;
            resp.audioData = (uint8_t*)ps_malloc(resp.audioLength);
            if (!resp.audioData) {
                resp.audioData = (uint8_t*)malloc(resp.audioLength);
            }
            if (resp.audioData) {
                size_t actual_len = 0;
                mbedtls_base64_decode(resp.audioData, resp.audioLength, &actual_len,
                    (const unsigned char*)audioB64.c_str(), audioB64.length());
                resp.audioLength = actual_len;
            }
        }
    }
    
    return resp;
}

ServerResponse CloudAPI::analyzeImage(camera_fb_t* fb) {
    Serial.println("[API] Sending image for danger analysis...");
    return sendImage(fb, "/analyze");
}

ServerResponse CloudAPI::describeImage(camera_fb_t* fb) {
    Serial.println("[API] Sending image for description...");
    return sendImage(fb, "/describe");
}

void CloudAPI::speakAlert(const String& message) {
    Serial.println("[ALERT] " + message);
    Audio.playAlertTone();
}
