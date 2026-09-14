#include "AppLogic.h"
#include "AudioMod.h"

AppLogic Logic;

// ─────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────
AppLogic::AppLogic()
    : _lastMessage(""),
      _currentPrompt(""),
      _lastPriority(PRIORITY_NORMAL),
      _lastAlertTimeP0(0),
      _lastAlertTimeP1(0),
      _lastAlertTimeP2(0),
      _alertCount(0),
      _alertIndex(0)
{
}

// ─────────────────────────────────────────────
// Cooldown management
// ─────────────────────────────────────────────
bool AppLogic::_isCoolingDown(HazardPriority priority, unsigned long nowMs) const {
    unsigned long lastTime = 0;
    unsigned long cooldown = 0;

    switch (priority) {
        case PRIORITY_P0:
            lastTime = _lastAlertTimeP0;
            cooldown = COOLDOWN_P0_MS;
            break;
        case PRIORITY_P1:
            lastTime = _lastAlertTimeP1;
            cooldown = COOLDOWN_P1_MS;
            break;
        case PRIORITY_P2:
            lastTime = _lastAlertTimeP2;
            cooldown = COOLDOWN_P2_MS;
            break;
        default:
            return false;
    }
    return (lastTime > 0) && (nowMs - lastTime < cooldown);
}

void AppLogic::_updateCooldown(HazardPriority priority, unsigned long nowMs) {
    switch (priority) {
        case PRIORITY_P0: _lastAlertTimeP0 = nowMs; break;
        case PRIORITY_P1: _lastAlertTimeP1 = nowMs; break;
        case PRIORITY_P2: _lastAlertTimeP2 = nowMs; break;
        default: break;
    }
}

// ─────────────────────────────────────────────
// processHazardResult — main hazard handler (PRD §4.1 / §4.2)
// ─────────────────────────────────────────────
void AppLogic::processHazardResult(const ServerResponse& resp) {
    if (!resp.success) return;

    unsigned long now = millis();

    if (!resp.hasAlert) {
        // Server says SAFE → absolute silence. Do not even log.
        Serial.println("[LOGIC] ✓ CLEAR — silence maintained.");
        return;
    }

    // Parse priority from server response (resp.priority set by CloudAPI)
    HazardPriority priority = (HazardPriority)resp.priority;

    // ── Cooldown gate ────────────────────────────────────────────────────────
    // P0 resets its own cooldown independently of P1/P2
    if (_isCoolingDown(priority, now)) {
        Serial.printf("[LOGIC] ⏱ %s cooling down — suppressed: %s\n",
                      priority == PRIORITY_P0 ? "P0" :
                      priority == PRIORITY_P1 ? "P1" : "P2",
                      resp.message.c_str());
        return;
    }

    // ── Priority arbitration ─────────────────────────────────────────────────
    // P0 preempts any currently-playing audio (the I2S driver buffers can be
    // flushed by feeding silence; simpler: P0 always wins immediately).
    if (priority == PRIORITY_P0) {
        Serial.printf("[LOGIC] ♦ P0 HEAD HAZARD — \"%s\"\n", resp.message.c_str());
        // Play earcon first
        Audio.playEarconP0();
        // Then WAV speech from server (if available), else the prompt is enough
        if (resp.audioData && resp.audioLength > 0) {
            Audio.feedWAVData(resp.audioData, resp.audioLength);
        }

    } else if (priority == PRIORITY_P1) {
        Serial.printf("[LOGIC] ◆ P1 FLOOR OBSTACLE — \"%s\"\n", resp.message.c_str());
        Audio.playEarconP1();
        if (resp.audioData && resp.audioLength > 0) {
            Audio.feedWAVData(resp.audioData, resp.audioLength);
        }

    } else if (priority == PRIORITY_P2) {
        Serial.printf("[LOGIC] ● P2 DYNAMIC — \"%s\"\n", resp.message.c_str());
        Audio.playEarconP2();
        if (resp.audioData && resp.audioLength > 0) {
            Audio.feedWAVData(resp.audioData, resp.audioLength);
        }
    }

    // Record cooldown and update state
    _updateCooldown(priority, now);
    _lastPriority  = priority;
    _lastMessage   = resp.message;
    _currentPrompt = resp.message;

    addAlert(resp.message, priority);
}

// Legacy compatibility shim
void AppLogic::processDangerResult(const ServerResponse& resp) {
    processHazardResult(resp);
}

// ─────────────────────────────────────────────
// processDescribeResult — single-press scene description
// ─────────────────────────────────────────────
void AppLogic::processDescribeResult(const ServerResponse& resp) {
    if (!resp.success) return;

    Serial.println("[LOGIC] 📝 DESCRIBE: " + resp.message);
    // No earcon for description — just speech
    if (resp.audioData && resp.audioLength > 0) {
        Audio.feedWAVData(resp.audioData, resp.audioLength);
    }

    _lastMessage = resp.message;
    addAlert(resp.message, PRIORITY_NORMAL);
}

// ─────────────────────────────────────────────
// processSearchResult — double-tap Object Locator
// ─────────────────────────────────────────────
void AppLogic::processSearchResult(const ServerResponse& resp) {
    if (!resp.success) return;

    if (!resp.hasAlert) {
        // Object not found — play "not found" speech
        Serial.println("[LOGIC] 🔍 Object not found.");
        if (resp.audioData && resp.audioLength > 0) {
            Audio.feedWAVData(resp.audioData, resp.audioLength);
        } else {
            // Fallback: short descending chirp to indicate "not found"
            Audio.playTone(660, 80, 0.3f);
            Audio.playTone(440, 120, 0.3f);
        }
        return;
    }

    Serial.println("[LOGIC] 🔍 FOUND: " + resp.message);

    // Play location speech
    if (resp.audioData && resp.audioLength > 0) {
        Audio.feedWAVData(resp.audioData, resp.audioLength);
    }

    // Play proximity click guidance
    // distanceNorm is packed into resp.elapsed field as a convention
    // (server sets it as the normalised distance 0.0–1.0)
    int clickFreqHz = getProximityClickFreq(resp.distanceNorm);
    Audio.playProximityClicks(clickFreqHz, 1500);

    _lastMessage = resp.message;
    addAlert(resp.message, PRIORITY_NORMAL);
}

// ─────────────────────────────────────────────
// addAlert
// ─────────────────────────────────────────────
void AppLogic::addAlert(const String& message, HazardPriority priority) {
    _alerts[_alertIndex].message   = message;
    _alerts[_alertIndex].timestamp = millis();
    _alerts[_alertIndex].priority  = priority;
    _alerts[_alertIndex].isDanger  = (priority >= PRIORITY_P1);
    _alertIndex = (_alertIndex + 1) % MAX_ALERTS;
    if (_alertCount < MAX_ALERTS) _alertCount++;
}

// ─────────────────────────────────────────────
// Getters
// ─────────────────────────────────────────────
String AppLogic::getLastMessage() const  { return _lastMessage;   }
String AppLogic::getCurrentPrompt() const { return _currentPrompt; }
HazardPriority AppLogic::getLastPriority() const { return _lastPriority; }

// ─────────────────────────────────────────────
// Proximity click frequency calculator
// distance_norm: 0.0 = very close, 1.0 = far
// Returns: 2 Hz (far) to 20 Hz (close)
// ─────────────────────────────────────────────
int AppLogic::getProximityClickFreq(float distanceNorm) const {
    distanceNorm = constrain(distanceNorm, 0.0f, 1.0f);
    return (int)(2.0f + (1.0f - distanceNorm) * 18.0f);
}

// ─────────────────────────────────────────────
// JSON string escaping — hazard labels/descriptions are free text and can
// contain quotes or control characters, which would otherwise produce
// malformed JSON and break the dashboard's fetch(...).json() calls.
// ─────────────────────────────────────────────
static String _jsonEscape(const String& s) {
    String out;
    out.reserve(s.length() + 8);
    for (size_t i = 0; i < s.length(); i++) {
        char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((uint8_t)c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ─────────────────────────────────────────────
// getStatusJSON — for web dashboard /status endpoint
// ─────────────────────────────────────────────
String AppLogic::getStatusJSON(SystemMode mode, bool serverOk) const {
    String modeStr;
    switch (mode) {
        case MODE_DETECTING:   modeStr = "detecting";   break;
        case MODE_DESCRIBING:  modeStr = "describing";  break;
        case MODE_LOCATING:    modeStr = "locating";    break;
        case MODE_STANDBY:     modeStr = "standby";     break;
        case MODE_LOW_POWER:   modeStr = "low_power";   break;
        default:               modeStr = "idle";        break;
    }

    String priStr;
    switch (_lastPriority) {
        case PRIORITY_P0: priStr = "P0"; break;
        case PRIORITY_P1: priStr = "P1"; break;
        case PRIORITY_P2: priStr = "P2"; break;
        default:          priStr = "CLEAR"; break;
    }

    String json = "{";
    json += "\"mode\":\""         + modeStr + "\",";
    json += "\"serverOk\":"       + String(serverOk ? "true" : "false") + ",";
    json += "\"lastMessage\":\""  + _jsonEscape(_lastMessage) + "\",";
    json += "\"lastPriority\":\""  + priStr + "\",";
    json += "\"alertCount\":"     + String(_alertCount) + ",";
    json += "\"uptime\":"         + String(millis() / 1000);
    json += "}";
    return json;
}

// ─────────────────────────────────────────────
// getAlertHistoryJSON — for web dashboard /alerts endpoint
// ─────────────────────────────────────────────
String AppLogic::getAlertHistoryJSON() const {
    String json = "[";
    int start = (_alertCount < MAX_ALERTS) ? 0 : _alertIndex;
    for (int i = 0; i < _alertCount; i++) {
        int idx = (start + i) % MAX_ALERTS;
        if (i > 0) json += ",";
        String priStr;
        switch (_alerts[idx].priority) {
            case PRIORITY_P0: priStr = "P0"; break;
            case PRIORITY_P1: priStr = "P1"; break;
            case PRIORITY_P2: priStr = "P2"; break;
            default:          priStr = "INFO"; break;
        }
        json += "{";
        json += "\"msg\":\""    + _jsonEscape(_alerts[idx].message) + "\",";
        json += "\"priority\":\"" + priStr + "\",";
        json += "\"danger\":"   + String(_alerts[idx].isDanger ? "true" : "false") + ",";
        json += "\"t\":"        + String(_alerts[idx].timestamp / 1000);
        json += "}";
    }
    json += "]";
    return json;
}
