#ifndef BATTERY_MOD_H
#define BATTERY_MOD_H

#include <Arduino.h>
#include "Config.h"

class BatteryMod {
public:
    BatteryMod();
    void begin();
    void update();
    int getPercentage();
    String getStatusString();
    bool isLowBattery();

private:
    int _percentage;
    unsigned long _lastCheckTime;
    const unsigned long _checkInterval = 10000; // Check every 10 seconds
    
    // Calibration values for 3.7V Li-Po
    const float _minVoltage = 3.2; // 0%
    const float _maxVoltage = 4.2; // 100%
};

extern BatteryMod Battery;

#endif
