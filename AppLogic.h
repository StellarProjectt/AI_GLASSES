#ifndef APP_LOGIC_H
#define APP_LOGIC_H

#include <Arduino.h>
#include "Config.h"
#include "CloudAPI.h"

// Alert log entry for web UI
struct AlertEntry {
    String message;
    unsigned long timestamp;
    bool isDanger;
};

class AppLogic {
public:
    AppLogic();
    
    // Process server response (danger detection)
    void processDangerResult(const ServerResponse& resp);
    
    // Process server response (description)
    void processDescribeResult(const ServerResponse& resp);
    
    // Add alert to log
    void addAlert(const String& message, bool isDanger);
    
    // Get detection status as JSON for web UI
    String getStatusJSON(SystemMode mode, bool serverOk);
    
    // Get alert history as JSON for web UI
    String getAlertHistoryJSON();
    
    // Get last message
    String getLastMessage();

private:
    String _lastMessage;
    unsigned long _lastAlertTime;
    
    // Alert history (circular buffer)
    static const int MAX_ALERTS = 20;
    AlertEntry _alerts[20];
    int _alertCount;
    int _alertIndex;
};

extern AppLogic Logic;

#endif
