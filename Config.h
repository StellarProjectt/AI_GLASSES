#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ==========================================
// 1. WiFi Configuration
// ==========================================
// Station Mode — Home WiFi credentials
extern const char* WIFI_SSID;
extern const char* WIFI_PASS;

// AP Mode Fallback (when WiFi unavailable)
#define WIFI_AP_SSID     "HomeSight_Demo"
#define WIFI_AP_PASS     "12345678"

// ==========================================
// 2. Python Server Configuration (LAN — no internet)
// ==========================================
// ⚠️ Change this to your computer's local IP.
//    Mac: System Settings → Network, or run `ifconfig | grep "inet "`
//    Windows: ipconfig
#define PYTHON_SERVER_IP    "192.168.0.102"
#define PYTHON_SERVER_PORT  5001

// ==========================================
// 3. Web Server Configuration
// ==========================================
#define WEB_SERVER_PORT  80

// ==========================================
// 4. Hardware Pins — XIAO ESP32-S3 Sense
// ==========================================

// I2S Speaker (MAX98357A or bone-conduction amp)
// Open-ear / bone-conduction transducers ONLY — PRD §5.2 forbids in-ear sealing
#define I2S_SPK_BCLK    7   // D8
#define I2S_SPK_LRC     8   // D9
#define I2S_SPK_DOUT    9   // D10

// I2S Microphone (for Object Locator voice query — double-tap mode)
#define I2S_MIC_SCK     -1
#define I2S_MIC_WS      -1
#define I2S_MIC_SD      -1

// Action Button (D0)
//   Single press  → MODE_DESCRIBING  (scene description)
//   Double tap    → MODE_LOCATING    (object locator, PRD §4.3)
#define PIN_ACTION_BTN  1
#define PIN_POWER_SW    2   // D1 — power switch

// Battery ADC
#define PIN_BATTERY_ADC 5   // D4 — voltage divider midpoint

// Camera (OV3660 on XIAO ESP32-S3 Sense)
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    10
#define SIOD_GPIO_NUM    40
#define SIOC_GPIO_NUM    39
#define Y9_GPIO_NUM      48
#define Y8_GPIO_NUM      11
#define Y7_GPIO_NUM      12
#define Y6_GPIO_NUM      14
#define Y5_GPIO_NUM      16
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM      17
#define Y2_GPIO_NUM      15
#define VSYNC_GPIO_NUM   38
#define HREF_GPIO_NUM    47
#define PCLK_GPIO_NUM    13

// ==========================================
// 5. HomeSight Hazard Detection — PRD §4.1
// ==========================================

// Auto detection interval (ms) — 1500 ms with local YOLO inference
#define AUTO_DETECT_INTERVAL_MS  1500

// Debounce for button presses
#define BUTTON_DEBOUNCE_MS       80

// Double-tap detection window (ms) — two presses within this = double-tap
#define DOUBLE_TAP_WINDOW_MS     400

// Hazard alert cooldown per priority class (ms)
// Prevents audio spam if the same hazard persists across frames
#define COOLDOWN_P0_MS           4000   // Head hazard: 4 s cooldown
#define COOLDOWN_P1_MS           6000   // Floor obstacle: 6 s cooldown
#define COOLDOWN_P2_MS           5000   // Dynamic object: 5 s cooldown

// PRD distance thresholds (sent to server for reference; server also validates)
#define P0_DISTANCE_THRESHOLD_M  1.2f   // Head hazard trigger distance
#define P1_DISTANCE_NEAR_M       0.8f   // Floor obstacle min distance
#define P1_DISTANCE_FAR_M        1.8f   // Floor obstacle max distance
#define P2_DISTANCE_THRESHOLD_M  1.5f   // Dynamic object trigger distance

// Battery
#define BATTERY_LOW_PERCENT      15
#define BATTERY_CHECK_INTERVAL_MS 10000

// ==========================================
// 6. System State Machine
// ==========================================
enum SystemMode {
    MODE_IDLE,
    MODE_STANDBY,
    MODE_DETECTING,      // Continuous hazard scan (default — every 1.5 s)
    MODE_DESCRIBING,     // Single-press scene description (one-shot)
    MODE_LOCATING,       // Double-tap Object Locator — PRD §4.3
    MODE_READING_OCR,    // Reserved (Phase 2)
    MODE_LOW_POWER
};

// ==========================================
// 7. Hazard Priority (mirrors Python hazard_classifier.py)
// ==========================================
enum HazardPriority {
    PRIORITY_NORMAL = 0,  // Silent — clear path
    PRIORITY_P2     = 1,  // Dynamic object (person / pet)
    PRIORITY_P1     = 2,  // Floor obstacle
    PRIORITY_P0     = 3   // Head-level critical — PREEMPTS ALL
};

// Audio earcon codes (returned by server)
#define EARCON_HIGH  "HIGH"   // P0 — two-tone rising chime
#define EARCON_LOW   "LOW"    // P1 — soft single pulse
#define EARCON_DIR   "DIR"    // P2 — directional triple blip

// ==========================================
// 8. Alert History
// ==========================================
#define MAX_ALERTS  20

#endif // CONFIG_H
