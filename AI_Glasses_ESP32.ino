#include <Arduino.h>
#include "Config.h"
#include "CameraMod.h"
#include "AudioMod.h"
#include "CloudAPI.h"
#include "AppLogic.h"
#include <WiFi.h>
#include <WebServer.h>
#include <mbedtls/base64.h>

// ==========================================
// Configuration — แก้ WiFi ตรงนี้
// ==========================================
const char* WIFI_SSID = "Bell_2.4G";
const char* WIFI_PASS = "22022927";
const char* GCP_API_KEY = "";

// ==========================================
// Global Variables
// ==========================================
WebServer server(WEB_SERVER_PORT);
SystemMode currentMode = MODE_DETECTING;
bool cameraReady = false;
bool speakerReady = false;
bool serverReady = false;

// Timing
unsigned long lastAutoDetectTime = 0;
unsigned long lastButtonPress = 0;
unsigned long lastServerCheck = 0;

// MJPEG streaming
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ==========================================
// Web Page HTML
// ==========================================
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Glasses - Detection System</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg: #0a0a0f;
            --card: #12121a;
            --border: #2a2a3e;
            --accent: #6c63ff;
            --danger: #ff4757;
            --success: #2ed573;
            --warning: #ffa502;
            --text: #e8e8f0;
            --text2: #8888a0;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle at 30% 50%, rgba(108,99,255,0.05) 0%, transparent 50%),
                        radial-gradient(circle at 70% 80%, rgba(255,71,87,0.03) 0%, transparent 50%);
            animation: bgShift 20s ease-in-out infinite;
            z-index: 0;
        }
        @keyframes bgShift {
            0%, 100% { transform: translate(0,0); }
            50% { transform: translate(-5%,-3%); }
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 16px;
            position: relative;
            z-index: 1;
        }
        
        .header {
            text-align: center;
            padding: 20px 0 16px;
        }
        .header h1 {
            font-size: 1.6em;
            font-weight: 700;
            background: linear-gradient(135deg, #6c63ff, #a855f7, #ff6b9d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p { color: var(--text2); font-size: 0.85em; margin-top: 4px; }
        
        .status-row {
            display: flex;
            gap: 8px;
            justify-content: center;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 12px;
            border-radius: 16px;
            font-size: 0.7em;
            font-weight: 500;
            border: 1px solid;
        }
        .badge.ok { background: rgba(46,213,115,0.1); border-color: rgba(46,213,115,0.3); color: var(--success); }
        .badge.err { background: rgba(255,71,87,0.1); border-color: rgba(255,71,87,0.3); color: var(--danger); }
        .badge.mode { background: rgba(108,99,255,0.1); border-color: rgba(108,99,255,0.3); color: var(--accent); }
        .dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }
        .dot.green { background: var(--success); }
        .dot.red { background: var(--danger); }
        .dot.blue { background: var(--accent); }
        @keyframes pulse {
            0%,100% { opacity:1; transform:scale(1); }
            50% { opacity:0.5; transform:scale(0.8); }
        }
        
        /* Camera */
        .stream-box {
            margin: 16px 0;
            border-radius: 14px;
            overflow: hidden;
            background: var(--card);
            border: 1px solid var(--border);
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        .stream-box.alert {
            border-color: var(--danger);
            box-shadow: 0 0 25px rgba(255,71,87,0.2);
        }
        .stream-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: rgba(255,255,255,0.02);
            border-bottom: 1px solid var(--border);
            font-size: 0.75em;
            color: var(--text2);
        }
        .live { display: flex; align-items: center; gap: 4px; color: var(--danger); font-weight: 600; }
        .stream-img {
            width: 100%;
            display: block;
            background: #000;
            min-height: 240px;
        }
        
        /* Describe Button */
        .btn-describe {
            display: block;
            width: 100%;
            padding: 14px;
            margin: 16px 0;
            border: none;
            border-radius: 12px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            background: linear-gradient(135deg, #6c63ff, #a855f7);
            color: white;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn-describe:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(108,99,255,0.4);
        }
        .btn-describe:active { transform: translateY(0); }
        .btn-describe:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Last Message */
        .message-box {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            margin: 16px 0;
            min-height: 60px;
        }
        .message-box h3 {
            font-size: 0.75em;
            color: var(--text2);
            margin-bottom: 8px;
        }
        .message-box .msg {
            font-size: 0.95em;
            line-height: 1.5;
        }
        .message-box .msg.danger { color: var(--danger); }
        
        /* Stats */
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 16px 0;
        }
        .stat {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }
        .stat-val {
            font-size: 1.4em;
            font-weight: 700;
            color: var(--accent);
        }
        .stat-val.danger { color: var(--danger); }
        .stat-val.ok { color: var(--success); }
        .stat-lbl {
            font-size: 0.65em;
            color: var(--text2);
            margin-top: 3px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Alert Log */
        .log-section {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            margin: 16px 0;
        }
        .log-section h3 {
            font-size: 0.8em;
            color: var(--text2);
            margin-bottom: 10px;
        }
        .log-list { max-height: 300px; overflow-y: auto; }
        .log-item {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 8px 10px;
            border-radius: 8px;
            margin-bottom: 5px;
            font-size: 0.78em;
            animation: slideIn 0.3s ease;
        }
        .log-item.danger { background: rgba(255,71,87,0.08); border: 1px solid rgba(255,71,87,0.15); }
        .log-item.info { background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.15); }
        .log-item .time { color: var(--text2); font-size: 0.85em; white-space: nowrap; margin-left: 8px; }
        @keyframes slideIn {
            from { opacity:0; transform:translateX(-8px); }
            to { opacity:1; transform:translateX(0); }
        }
        .no-logs { text-align: center; color: var(--text2); font-size: 0.8em; padding: 20px; }
        
        /* Audio Player */
        .audio-section {
            margin: 16px 0;
        }
        .audio-section audio {
            width: 100%;
            border-radius: 8px;
        }
        
        .footer {
            text-align: center;
            padding: 16px;
            color: var(--text2);
            font-size: 0.65em;
        }
        
        @media (max-width: 600px) {
            .header h1 { font-size: 1.3em; }
            .stats { gap: 6px; }
            .stat { padding: 10px 6px; }
            .stat-val { font-size: 1.1em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Glasses</h1>
            <p>Object Detection + Voice Alert System</p>
            <div class="status-row">
                <div class="badge ok" id="wifiBadge"><div class="dot green"></div> WiFi</div>
                <div class="badge err" id="serverBadge"><div class="dot red"></div> Server</div>
                <div class="badge mode" id="modeBadge"><div class="dot blue"></div> Detecting</div>
            </div>
        </div>
        
        <div class="stream-box" id="streamBox">
            <div class="stream-top">
                <span>📷 OV3660 Camera</span>
                <div class="live"><div class="dot red"></div> LIVE</div>
            </div>
            <img class="stream-img" id="stream" src="/stream" alt="Camera">
        </div>
        
        <button class="btn-describe" id="btnDescribe" onclick="describeScene()">
            🎤 อธิบายสิ่งที่อยู่ตรงหน้า
        </button>
        
        <div class="message-box" id="messageBox">
            <h3>💬 ข้อความล่าสุด</h3>
            <div class="msg" id="lastMsg">รอการตรวจจับ...</div>
        </div>
        
        <div class="audio-section" id="audioSection" style="display:none;">
            <audio id="audioPlayer" controls autoplay></audio>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-val" id="alertCount">0</div>
                <div class="stat-lbl">Alerts</div>
            </div>
            <div class="stat">
                <div class="stat-val ok" id="uptime">0s</div>
                <div class="stat-lbl">Uptime</div>
            </div>
            <div class="stat">
                <div class="stat-val" id="serverStatus">--</div>
                <div class="stat-lbl">Server</div>
            </div>
        </div>
        
        <div class="log-section">
            <h3>📋 Alert & Description Log</h3>
            <div class="log-list" id="logList">
                <div class="no-logs">ยังไม่มีการแจ้งเตือน...</div>
            </div>
        </div>
        
        <div class="footer">
            AI Glasses • ESP32-S3 + OV3660 + Gemini Flash + gTTS
        </div>
    </div>
    
    <script>
        function updateStatus() {
            fetch('/status').then(r => r.json()).then(data => {
                // Mode badge
                const modeBadge = document.getElementById('modeBadge');
                const modeText = data.mode === 'detecting' ? 'Detecting' : 
                                 data.mode === 'describing' ? 'Describing...' : 'Idle';
                modeBadge.innerHTML = '<div class="dot blue"></div> ' + modeText;
                
                // Server badge
                const sb = document.getElementById('serverBadge');
                if (data.serverOk) {
                    sb.className = 'badge ok';
                    sb.innerHTML = '<div class="dot green"></div> Server OK';
                    document.getElementById('serverStatus').textContent = '✅';
                } else {
                    sb.className = 'badge err';
                    sb.innerHTML = '<div class="dot red"></div> Server ❌';
                    document.getElementById('serverStatus').textContent = '❌';
                }
                
                // Last message
                if (data.lastMessage) {
                    document.getElementById('lastMsg').textContent = data.lastMessage;
                }
                
                // Uptime
                const m = Math.floor(data.uptime / 60);
                const s = data.uptime % 60;
                document.getElementById('uptime').textContent = m > 0 ? m+'m '+s+'s' : s+'s';
                
                // Alert count
                document.getElementById('alertCount').textContent = data.alertCount;
            }).catch(() => {});
        }
        
        function updateAlerts() {
            fetch('/alerts').then(r => r.json()).then(alerts => {
                const el = document.getElementById('logList');
                if (!alerts.length) {
                    el.innerHTML = '<div class="no-logs">ยังไม่มีการแจ้งเตือน...</div>';
                    return;
                }
                let html = '';
                for (let i = alerts.length - 1; i >= 0; i--) {
                    const a = alerts[i];
                    const cls = a.danger ? 'danger' : 'info';
                    const icon = a.danger ? '⚠️' : '📝';
                    const m = Math.floor(a.t / 60);
                    const s = a.t % 60;
                    const ts = m > 0 ? m+'m '+s+'s' : s+'s';
                    html += '<div class="log-item '+cls+'">';
                    html += '<span>'+icon+' '+a.msg+'</span>';
                    html += '<span class="time">'+ts+'</span>';
                    html += '</div>';
                }
                el.innerHTML = html;
            }).catch(() => {});
        }
        
        function describeScene() {
            const btn = document.getElementById('btnDescribe');
            btn.disabled = true;
            btn.textContent = '⏳ กำลังวิเคราะห์...';
            
            fetch('/trigger-describe', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    btn.disabled = false;
                    btn.textContent = '🎤 อธิบายสิ่งที่อยู่ตรงหน้า';
                    
                    if (data.message) {
                        document.getElementById('lastMsg').textContent = data.message;
                    }
                    
                    // Play audio if available
                    if (data.audio_base64) {
                        const audio = document.getElementById('audioPlayer');
                        audio.src = 'data:audio/mp3;base64,' + data.audio_base64;
                        document.getElementById('audioSection').style.display = 'block';
                        audio.play();
                    }
                    
                    updateAlerts();
                })
                .catch(() => {
                    btn.disabled = false;
                    btn.textContent = '🎤 อธิบายสิ่งที่อยู่ตรงหน้า';
                });
        }
        
        setInterval(updateStatus, 1000);
        setInterval(updateAlerts, 3000);
        updateStatus();
        updateAlerts();
    </script>
</body>
</html>
)rawliteral";

// ==========================================
// Web Server Handlers
// ==========================================
void handleRoot() {
    server.send(200, "text/html", INDEX_HTML);
}

void handleStream() {
    WiFiClient client = server.client();
    
    String response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: " + String(_STREAM_CONTENT_TYPE) + "\r\n";
    response += "Access-Control-Allow-Origin: *\r\n";
    response += "\r\n";
    client.print(response);
    
    while (client.connected()) {
        camera_fb_t* fb = Camera.captureFrame();
        if (!fb) {
            Serial.println("Stream: capture failed, retrying...");
            delay(200);
            continue;
        }
        
        char partBuf[64];
        snprintf(partBuf, sizeof(partBuf), _STREAM_PART, fb->len);
        
        client.print(_STREAM_BOUNDARY);
        client.print(partBuf);
        client.write(fb->buf, fb->len);
        
        Camera.releaseFrame(fb);
        delay(80); // ~12 FPS
    }
}

void handleStatus() {
    server.send(200, "application/json", Logic.getStatusJSON(currentMode, serverReady));
}

void handleAlerts() {
    server.send(200, "application/json", Logic.getAlertHistoryJSON());
}

// Trigger scene description from web UI button
void handleTriggerDescribe() {
    if (!cameraReady) {
        server.send(500, "application/json", "{\"error\":\"Camera not ready\"}");
        return;
    }
    
    Serial.println("\n[WEB] Describe button pressed!");
    currentMode = MODE_DESCRIBING;
    
    camera_fb_t* fb = Camera.captureFrame();
    if (!fb) {
        server.send(500, "application/json", "{\"error\":\"Capture failed\"}");
        currentMode = MODE_DETECTING;
        return;
    }
    
    ServerResponse resp = Cloud.describeImage(fb);
    Camera.releaseFrame(fb);
    
    currentMode = MODE_DETECTING;
    
    if (resp.success) {
        Logic.processDescribeResult(resp);
        
        // Build response with audio if available
        String json = "{";
        json += "\"message\":\"" + resp.message + "\"";
        
        if (resp.audioData && resp.audioLength > 0) {
            // Re-encode to base64 for the web UI using mbedtls
            size_t b64_len = ((resp.audioLength + 2) / 3) * 4 + 1;
            char* b64_buf = (char*)malloc(b64_len);
            if (b64_buf) {
                size_t actual_len = 0;
                mbedtls_base64_encode((unsigned char*)b64_buf, b64_len, &actual_len, resp.audioData, resp.audioLength);
                b64_buf[actual_len] = '\0';
                json += ",\"audio_base64\":\"" + String(b64_buf) + "\"";
                free(b64_buf);
            }
            free(resp.audioData);
        }
        
        json += ",\"elapsed\":" + String(resp.elapsed);
        json += "}";
        
        server.send(200, "application/json", json);
    } else {
        server.send(500, "application/json", "{\"error\":\"Server analysis failed\"}");
    }
}

// ==========================================
// Setup
// ==========================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println();
    Serial.println("╔══════════════════════════════════════════╗");
    Serial.println("║   AI Glasses - Object Detection System   ║");
    Serial.println("║   Camera: OV3660 | AI: Gemini Flash      ║");
    Serial.println("╚══════════════════════════════════════════╝");
    Serial.println();
    
    // Initialize Button
    pinMode(PIN_ACTION_BTN, INPUT_PULLUP);
    
    // Initialize Speaker (optional)
    speakerReady = Audio.begin();
    Serial.println(speakerReady ? "[OK] Speaker" : "[--] Speaker not available");
    
    // Initialize Camera
    Serial.println("Initializing OV3660 camera...");
    cameraReady = Camera.begin();
    Serial.println(cameraReady ? "[OK] Camera ready!" : "[FAIL] Camera failed!");
    
    // Connect to WiFi
    if (Cloud.connectWiFi()) {
        Serial.println("[OK] WiFi connected");
    } else {
        Serial.println("[!!] WiFi failed, starting AP mode...");
        Cloud.startAP();
    }
    
    String ip = Cloud.getIP();
    
    // Check Python server
    Serial.print("Checking Python server... ");
    serverReady = Cloud.checkServer();
    Serial.println(serverReady ? "OK!" : "NOT FOUND");
    
    if (!serverReady) {
        Serial.println();
        Serial.println("⚠️  Python server ไม่พร้อม!");
        Serial.println("   1. cd python_server");
        Serial.println("   2. pip install -r requirements.txt");
        Serial.println("   3. python server.py");
        Serial.printf( "   4. แก้ PYTHON_SERVER_IP ใน Config.h เป็น IP คอมของคุณ\n");
        Serial.println("   (ระบบจะลองเชื่อมต่อใหม่ทุก 10 วินาที)");
    }
    
    Serial.println();
    Serial.println("╔══════════════════════════════════════════╗");
    Serial.printf( "║  URL:  http://%-26s ║\n", ip.c_str());
    Serial.printf( "║  Server: %-31s ║\n", serverReady ? "✅ Connected" : "❌ Not found");
    Serial.println("╚══════════════════════════════════════════╝");
    Serial.println();
    
    // Setup Web Server
    server.on("/", HTTP_GET, handleRoot);
    server.on("/stream", HTTP_GET, handleStream);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/alerts", HTTP_GET, handleAlerts);
    server.on("/trigger-describe", HTTP_POST, handleTriggerDescribe);
    
    server.begin();
    Serial.println("[OK] Web server started");
    Serial.println("กดปุ่ม D0 เพื่อถามว่ามีอะไรข้างหน้า");
    Serial.println();
}

// ==========================================
// Main Loop
// ==========================================
void loop() {
    server.handleClient();
    
    unsigned long now = millis();
    
    // ─── Check Button Press ───
    if (digitalRead(PIN_ACTION_BTN) == LOW && now - lastButtonPress > BUTTON_DEBOUNCE_MS) {
        lastButtonPress = now;
        
        Serial.println("\n🔘 Button pressed! Describing scene...");
        currentMode = MODE_DESCRIBING;
        
        if (speakerReady) {
            Audio.playTone(660, 100); // Short beep to confirm button press
        }
        
        if (cameraReady && serverReady) {
            camera_fb_t* fb = Camera.captureFrame();
            if (fb) {
                ServerResponse resp = Cloud.describeImage(fb);
                Camera.releaseFrame(fb);
                
                if (resp.success) {
                    Logic.processDescribeResult(resp);
                    Serial.println("📝 " + resp.message);
                    
                    if (resp.audioData) {
                        free(resp.audioData); // We played alert tone already
                    }
                } else {
                    Serial.println("❌ Server analysis failed");
                    Logic.addAlert("Server ไม่ตอบสนอง", true);
                }
            }
        } else {
            Serial.println("❌ Camera or Server not ready");
        }
        
        currentMode = MODE_DETECTING;
    }
    
    // ─── Auto Danger Detection (every 3 seconds) ───
    if (cameraReady && serverReady && currentMode == MODE_DETECTING 
        && now - lastAutoDetectTime > AUTO_DETECT_INTERVAL_MS) {
        lastAutoDetectTime = now;
        
        camera_fb_t* fb = Camera.captureFrame();
        if (fb) {
            ServerResponse resp = Cloud.analyzeImage(fb);
            Camera.releaseFrame(fb);
            
            if (resp.success) {
                Logic.processDangerResult(resp);
                
                if (resp.audioData) {
                    free(resp.audioData);
                }
            }
        }
    }
    
    // ─── Retry Server Connection (every 10 seconds) ───
    if (!serverReady && now - lastServerCheck > 10000) {
        lastServerCheck = now;
        serverReady = Cloud.checkServer();
        if (serverReady) {
            Serial.println("✅ Python server connected!");
            Logic.addAlert("เชื่อมต่อ AI Server สำเร็จ", false);
        }
    }
    
    delay(1);
}
