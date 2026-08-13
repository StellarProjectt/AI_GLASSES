#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ==========================================
// 1. WiFi Configuration
// ==========================================
// Station Mode — เชื่อมต่อ WiFi บ้าน
extern const char* WIFI_SSID;
extern const char* WIFI_PASS;
extern const char* GCP_API_KEY;

// AP Mode Fallback
#define WIFI_AP_SSID     "AI_Glasses_Demo"
#define WIFI_AP_PASS     "12345678"

// ==========================================
// 2. Python Server Configuration
// ==========================================
// ⚠️ เปลี่ยน IP นี้เป็น IP ของคอมที่รัน python server
// ดู IP ได้จาก: ifconfig (Mac/Linux) หรือ ipconfig (Windows)
#define PYTHON_SERVER_IP    "192.168.0.102"
#define PYTHON_SERVER_PORT  5001

// ==========================================
// 3. Web Server Configuration
// ==========================================
#define WEB_SERVER_PORT  80

// ==========================================
// 4. Hardware Pins Configuration (XIAO ESP32-S3 Sense)
// ==========================================
// I2S Speaker (MAX98357A)
#define I2S_SPK_BCLK    7   // D8
#define I2S_SPK_LRC     8   // D9
#define I2S_SPK_DOUT    9   // D10

// I2S Microphone — ยังไม่ใช้
#define I2S_MIC_SCK     -1
#define I2S_MIC_WS      -1
#define I2S_MIC_SD      -1

// Button & Switch
#define PIN_ACTION_BTN  1   // D0 — ปุ่มกดถามคำถาม "มีอะไรข้างหน้า?"
#define PIN_POWER_SW    2   // D1 — สวิตช์เปิด-ปิด

// Battery ADC
#define PIN_BATTERY_ADC 5   // D4 — ขาอ่านแรงดันแบตเตอรี่

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
// 5. Detection Configuration
// ==========================================
#define AUTO_DETECT_INTERVAL_MS  6000  // ส่งภาพไป Python ทุก 6 วินาที (ป้องกัน Quota Limit)
#define BUTTON_DEBOUNCE_MS       500   // ป้องกันกดปุ่มซ้ำ

// Motion detection (ยังใช้อยู่เพื่อลด API calls)
#define MOTION_THRESHOLD       15
#define MOTION_PERCENT_ALERT   8.0f
#define ALERT_COOLDOWN_MS      6000
#define DETECTION_FRAME_WIDTH  160
#define DETECTION_FRAME_HEIGHT 120

// ==========================================
// 6. System States & Constants
// ==========================================
enum SystemMode {
    MODE_IDLE,
    MODE_STANDBY,
    MODE_DETECTING,       // ตรวจอันตรายอัตโนมัติ
    MODE_DESCRIBING,      // กำลังอธิบายภาพ (กดปุ่ม)
    MODE_READING_OCR,
    MODE_DESCRIBE_SCENE,
    MODE_SEARCHING,
    MODE_LOW_POWER
};

enum DistanceLevel {
    DIST_SAFE,
    DIST_PREPARE,
    DIST_WARNING,
    DIST_DANGER
};

#endif // CONFIG_H
