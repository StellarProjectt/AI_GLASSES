#include "CloudAPI.h"
#include "Config.h"
#include "AudioMod.h"
#include <ArduinoJson.h>
#include <mbedtls/base64.h>

CloudAPI Cloud;

CloudAPI::CloudAPI() {
    _apMode    = false;
    _serverUrl = "http://" + String(PYTHON_SERVER_IP) + ":" + String(PYTHON_SERVER_PORT);
}

// ─────────────────────────────────────────────
// WiFi
// ─────────────────────────────────────────────
bool CloudAPI::connectWiFi() {
    Serial.print("[WiFi] Connecting to: ");
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
        Serial.print("[WiFi] Connected — IP: ");
        Serial.println(WiFi.localIP());
        return true;
    }
    Serial.println("[WiFi] Connection failed.");
    return false;
}

bool CloudAPI::startAP() {
    Serial.println("[WiFi] Starting AP mode...");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASS);
    delay(500);
    _apMode = true;
    Serial.print("[WiFi] AP IP: ");
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

// ─────────────────────────────────────────────
// Server health check
// ─────────────────────────────────────────────
bool CloudAPI::checkServer() {
    if (!isConnected()) return false;
    HTTPClient http;
    http.setTimeout(3000);
    http.begin(_serverUrl + "/health");
    int code = http.GET();
    http.end();
    return code == 200;
}

// ─────────────────────────────────────────────
// Audio decode helper
// ─────────────────────────────────────────────
/*static*/ uint8_t* CloudAPI::_decodeAudioField(const String& b64, size_t& outLen) {
    if (b64.length() == 0) { outLen = 0; return nullptr; }

    size_t expectedLen = 0;
    mbedtls_base64_decode(nullptr, 0, &expectedLen,
        (const unsigned char*)b64.c_str(), b64.length());

    uint8_t* buf = (uint8_t*)ps_malloc(expectedLen);
    if (!buf) buf = (uint8_t*)malloc(expectedLen);
    if (!buf) { outLen = 0; return nullptr; }

    size_t actualLen = 0;
    mbedtls_base64_decode(buf, expectedLen, &actualLen,
        (const unsigned char*)b64.c_str(), b64.length());
    outLen = actualLen;
    return buf;
}

// ─────────────────────────────────────────────
// Core: send JPEG frame as multipart POST
// ─────────────────────────────────────────────
ServerResponse CloudAPI::sendImage(
    camera_fb_t* fb,
    const String& endpoint,
    const String& extraField,
    const String& extraValue
) {
    ServerResponse resp = {false, false, "", "", 0, "", 0.0f, nullptr, 0, nullptr, 0, 0.0f};
    if (!isConnected() || !fb) return resp;

    HTTPClient http;
    http.setTimeout(8000);  // 8 s — local LAN should respond in < 300 ms normally
    http.begin(_serverUrl + endpoint);

    String boundary = "HomeSightBoundary42";
    http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

    // ── Build multipart body ─────────────────────────────────────────────────
    String partImage  = "--" + boundary + "\r\n"
                        "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n"
                        "Content-Type: image/jpeg\r\n\r\n";

    String partExtra = "";
    if (extraField.length() > 0) {
        partExtra = "\r\n--" + boundary + "\r\n"
                    "Content-Disposition: form-data; name=\"" + extraField + "\"\r\n\r\n"
                    + extraValue;
    }

    String bodyEnd = "\r\n--" + boundary + "--\r\n";

    size_t totalLen = partImage.length() + fb->len + partExtra.length() + bodyEnd.length();
    uint8_t* body   = (uint8_t*)ps_malloc(totalLen);
    if (!body) body = (uint8_t*)malloc(totalLen);
    if (!body) {
        Serial.println("[API] Allocation failed.");
        http.end();
        return resp;
    }

    size_t off = 0;
    memcpy(body + off, partImage.c_str(), partImage.length()); off += partImage.length();
    memcpy(body + off, fb->buf, fb->len);                      off += fb->len;
    if (partExtra.length()) {
        memcpy(body + off, partExtra.c_str(), partExtra.length()); off += partExtra.length();
    }
    memcpy(body + off, bodyEnd.c_str(), bodyEnd.length());

    // ── POST ─────────────────────────────────────────────────────────────────
    int httpCode = http.POST(body, totalLen);
    free(body);

    if (httpCode != 200) {
        Serial.printf("[API] HTTP %d on %s\n", httpCode, endpoint.c_str());
        http.end();
        return resp;
    }

    // ── Parse JSON response ───────────────────────────────────────────────────
    String payload = http.getString();
    http.end();

    DynamicJsonDocument doc(16384);
    DeserializationError err = deserializeJson(doc, payload);
    if (err) {
        Serial.printf("[API] JSON parse error: %s\n", err.c_str());
        return resp;
    }

    resp.success = true;
    resp.elapsed = doc["elapsed"] | 0.0f;

    // /analyze response
    if (doc.containsKey("alert")) {
        resp.hasAlert = doc["alert"].as<bool>();
        if (resp.hasAlert) {
            resp.message     = doc["prompt"]   | doc["message"]  | String("");
            resp.earconCode  = doc["earcon"]   | String("");
            resp.side        = doc["side"]     | String("center");
            resp.distanceNorm= doc["dist_m"]   | 1.0f;
            // Map priority string to int
            String priStr    = doc["priority"] | String("NORMAL");
            if      (priStr == "P0") resp.priority = PRIORITY_P0;
            else if (priStr == "P1") resp.priority = PRIORITY_P1;
            else if (priStr == "P2") resp.priority = PRIORITY_P2;
            else                     resp.priority = PRIORITY_NORMAL;
        }
    }

    // /describe response
    if (doc.containsKey("description")) {
        resp.message  = doc["description"].as<String>();
        resp.hasAlert = false;
    }

    // /search response
    if (doc.containsKey("found")) {
        resp.hasAlert = doc["found"].as<bool>();
        resp.message  = doc["message"] | String("Not found.");
        resp.distanceNorm = doc["distance_norm"] | 1.0f;
        // Proximity click audio (separate field)
        if (doc.containsKey("click_audio_b64")) {
            resp.clickAudioData = _decodeAudioField(
                doc["click_audio_b64"].as<String>(), resp.clickAudioLength);
        }
    }

    // Main audio (speech + earcon combined WAV)
    if (doc.containsKey("audio_base64")) {
        resp.audioData = _decodeAudioField(
            doc["audio_base64"].as<String>(), resp.audioLength);
    }

    return resp;
}

// ─────────────────────────────────────────────
// Public API methods
// ─────────────────────────────────────────────
ServerResponse CloudAPI::analyzeImage(camera_fb_t* fb) {
    Serial.println("[API] → /analyze");
    return sendImage(fb, "/analyze");
}

ServerResponse CloudAPI::describeImage(camera_fb_t* fb) {
    Serial.println("[API] → /describe");
    return sendImage(fb, "/describe");
}

ServerResponse CloudAPI::searchObject(camera_fb_t* fb, const String& query) {
    Serial.println("[API] → /search : " + query);
    return sendImage(fb, "/search", "query", query);
}

void CloudAPI::speakAlert(const String& message) {
    Serial.println("[ALERT] " + message);
    Audio.playEarconP0();  // Fallback when no server
}
