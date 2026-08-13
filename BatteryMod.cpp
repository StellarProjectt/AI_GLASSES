#include "BatteryMod.h"

BatteryMod Battery;

BatteryMod::BatteryMod() {
    _percentage = 100;
    _lastCheckTime = 0;
}

void BatteryMod::begin() {
    pinMode(PIN_BATTERY_ADC, INPUT);
    // Initial check
    update();
}

void BatteryMod::update() {
    if (millis() - _lastCheckTime > _checkInterval) {
        _lastCheckTime = millis();
        
        // Read ADC value (0-4095 for ESP32)
        int adcValue = analogRead(PIN_BATTERY_ADC);
        
        // Convert ADC to voltage (assuming a voltage divider 100k/100k)
        // Adjust these calculations based on the actual voltage divider used.
        float voltage = (adcValue / 4095.0) * 3.3 * 2.0; 
        
        if (voltage >= _maxVoltage) {
            _percentage = 100;
        } else if (voltage <= _minVoltage) {
            _percentage = 0;
        } else {
            _percentage = (int)(((voltage - _minVoltage) / (_maxVoltage - _minVoltage)) * 100);
        }
    }
}

int BatteryMod::getPercentage() {
    return _percentage;
}

bool BatteryMod::isLowBattery() {
    return _percentage < 15;
}

String BatteryMod::getStatusString() {
    if (_percentage > 20) return "แบตเตอรี่ปรกติ";
    if (_percentage > 10) return "แบตเตอรี่เหลือน้อย กรุณาเตรียมเปลี่ยนแบตเตอรี่";
    if (_percentage > 5)  return "แบตเตอรี่ใกล้หมด กรุณาเปลี่ยนแบตเตอรี่โดยเร็ว";
    return "แบตเตอรี่ต่ำมาก ระบบจะปิดการทำงานในไม่ช้า";
}
