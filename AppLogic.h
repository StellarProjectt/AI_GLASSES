#ifndef APP_LOGIC_H
#define APP_LOGIC_H

#include <Arduino.h>
#include "Config.h"
#include "CloudAPI.h"

// ─────────────────────────────────────────────
// Alert log entry (for web UI history)
// ─────────────────────────────────────────────
struct AlertEntry {
    String         message;
    unsigned long  timestamp;
    HazardPriority priority;
    bool           isDanger;   // kept for backwards-compat with web UI
};

// ─────────────────────────────────────────────
// Double-tap detector state
// ─────────────────────────────────────────────
struct DoubleTapDetector {
    unsigned long lastPressMs  = 0;
    int           pressCount   = 0;
    bool          waitingSecond = false;

    // Call this on every button-down edge. Returns true when a double-tap
    // is confirmed (second press within DOUBLE_TAP_WINDOW_MS).
    bool update(unsigned long nowMs) {
        if (waitingSecond &&
            nowMs - lastPressMs <= DOUBLE_TAP_WINDOW_MS) {
            // Second tap confirmed
            pressCount    = 0;
            waitingSecond = false;
            return true;
        }
        // First tap — start window
        lastPressMs   = nowMs;
        waitingSecond = true;
        pressCount    = 1;
        return false;
    }

    // Returns true if the double-tap window has expired (single-tap intent)
    bool windowExpired(unsigned long nowMs) const {
        return waitingSecond &&
               (nowMs - lastPressMs > DOUBLE_TAP_WINDOW_MS);
    }
};


// ─────────────────────────────────────────────
// AppLogic
// ─────────────────────────────────────────────
class AppLogic {
public:
    AppLogic();

    // ── HomeSight v3: Priority-aware hazard processing ──────────────────────
    // Called every AUTO_DETECT_INTERVAL_MS with the server's /analyze response.
    // Implements:
    //   • P0 preempts all lower-priority audio
    //   • Per-priority cooldown gates (prevent alert spam)
    //   • Earcon selection based on priority class
    void processHazardResult(const ServerResponse& resp);

    // Legacy entry point for backward compatibility (maps to processHazardResult)
    void processDangerResult(const ServerResponse& resp);

    // Called after a single-press /describe response
    void processDescribeResult(const ServerResponse& resp);

    // Called after a double-tap /search response
    void processSearchResult(const ServerResponse& resp);

    // ── Alert history & UI ───────────────────────────────────────────────────
    void   addAlert(const String& message, HazardPriority priority = PRIORITY_NORMAL);
    String getLastMessage()  const;
    String getCurrentPrompt() const;
    HazardPriority getLastPriority() const;

    // JSON helpers for web dashboard
    String getStatusJSON(SystemMode mode, bool serverOk) const;
    String getAlertHistoryJSON() const;

    // ── Proximity click tone for Object Locator ──────────────────────────────
    // Returns earcon frequency (Hz) that increases as distance_norm → 0.
    // distance_norm: 0.0 = object very close, 1.0 = far away.
    int getProximityClickFreq(float distance_norm) const;

    // ── Double-tap detector (call from main loop on button edge) ────────────
    DoubleTapDetector doubleTap;

private:
    String         _lastMessage;
    String         _currentPrompt;
    HazardPriority _lastPriority;

    // Per-priority cooldown tracking
    unsigned long _lastAlertTimeP0;
    unsigned long _lastAlertTimeP1;
    unsigned long _lastAlertTimeP2;

    // Alert history circular buffer
    AlertEntry _alerts[MAX_ALERTS];
    int        _alertCount;
    int        _alertIndex;

    bool _isCoolingDown(HazardPriority priority, unsigned long nowMs) const;
    void _updateCooldown(HazardPriority priority, unsigned long nowMs);
};

extern AppLogic Logic;

#endif // APP_LOGIC_H
