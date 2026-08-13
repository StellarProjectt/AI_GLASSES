#include "AppLogic.h"
#include "AudioMod.h"

AppLogic Logic;

AppLogic::AppLogic() {
    _lastMessage = "";
    _lastAlertTime = 0;
    _alertCount = 0;
    _alertIndex = 0;
}

void AppLogic::processDangerResult(const ServerResponse& resp) {
    if (!resp.success) return;
    
    if (resp.hasAlert) {
        Serial.println("[DANGER] " + resp.message);
        addAlert(resp.message, true);
        
        // Play alert tone via speaker
        Audio.playAlertTone();
        
        _lastMessage = resp.message;
        _lastAlertTime = millis();
    }
}

void AppLogic::processDescribeResult(const ServerResponse& resp) {
    if (!resp.success) return;
    
    Serial.println("[DESCRIBE] " + resp.message);
    addAlert(resp.message, false);
    _lastMessage = resp.message;
}

void AppLogic::addAlert(const String& message, bool isDanger) {
    _alerts[_alertIndex].message = message;
    _alerts[_alertIndex].timestamp = millis();
    _alerts[_alertIndex].isDanger = isDanger;
    _alertIndex = (_alertIndex + 1) % MAX_ALERTS;
    if (_alertCount < MAX_ALERTS) _alertCount++;
}

String AppLogic::getLastMessage() {
    return _lastMessage;
}

String AppLogic::getStatusJSON(SystemMode mode, bool serverOk) {
    String modeStr;
    switch(mode) {
        case MODE_DETECTING: modeStr = "detecting"; break;
        case MODE_DESCRIBING: modeStr = "describing"; break;
        case MODE_STANDBY: modeStr = "standby"; break;
        default: modeStr = "idle"; break;
    }
    
    String json = "{";
    json += "\"mode\":\"" + modeStr + "\",";
    json += "\"serverOk\":" + String(serverOk ? "true" : "false") + ",";
    json += "\"lastMessage\":\"" + _lastMessage + "\",";
    json += "\"alertCount\":" + String(_alertCount) + ",";
    json += "\"uptime\":" + String(millis() / 1000);
    json += "}";
    return json;
}

String AppLogic::getAlertHistoryJSON() {
    String json = "[";
    int start = (_alertCount < MAX_ALERTS) ? 0 : _alertIndex;
    for (int i = 0; i < _alertCount; i++) {
        int idx = (start + i) % MAX_ALERTS;
        if (i > 0) json += ",";
        json += "{";
        json += "\"msg\":\"" + _alerts[idx].message + "\",";
        json += "\"danger\":" + String(_alerts[idx].isDanger ? "true" : "false") + ",";
        json += "\"t\":" + String(_alerts[idx].timestamp / 1000);
        json += "}";
    }
    json += "]";
    return json;
}
