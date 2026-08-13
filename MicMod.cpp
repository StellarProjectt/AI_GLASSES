#include "MicMod.h"

MicMod Mic;

MicMod::MicMod() {
    _isInitialized = false;
}

bool MicMod::begin() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000, // 16kHz is standard for Speech-to-Text
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .mck_io_num = I2S_PIN_NO_CHANGE,
        .bck_io_num = I2S_MIC_SCK,
        .ws_io_num = I2S_MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SD
    };

    esp_err_t err = i2s_driver_install(I2S_PORT_MIC, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("Failed installing I2S driver for mic: %d\n", err);
        return false;
    }

    err = i2s_set_pin(I2S_PORT_MIC, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("Failed setting I2S pins for mic: %d\n", err);
        return false;
    }

    _isInitialized = true;
    return true;
}

size_t MicMod::readAudio(uint8_t* buffer, size_t bufferSize) {
    if (!_isInitialized) return 0;
    size_t bytesRead = 0;
    i2s_read(I2S_PORT_MIC, buffer, bufferSize, &bytesRead, portMAX_DELAY);
    return bytesRead;
}
