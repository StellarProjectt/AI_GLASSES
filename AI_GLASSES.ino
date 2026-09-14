#include <Arduino.h>
#include "Config.h"
#include "CameraMod.h"
#include "AudioMod.h"
#include "CloudAPI.h"
#include "AppLogic.h"
#include "BatteryMod.h"
#include <WiFi.h>
#include <WebServer.h>
#include <mbedtls/base64.h>

// ==========================================
// WiFi Credentials — change these
// ==========================================
const char* WIFI_SSID = "Stellar";
const char* WIFI_PASS = "11Stellar";

// ==========================================
// Global State
// ==========================================
WebServer    server(WEB_SERVER_PORT);
SystemMode   currentMode   = MODE_DETECTING;
bool         cameraReady   = false;
bool         speakerReady  = false;
bool         serverReady   = false;

// Timing
unsigned long lastAutoDetectTime  = 0;
unsigned long lastServerCheck     = 0;
unsigned long lastBatteryCheck    = 0;
unsigned long lastBatteryAlertMs  = 0;

// Button edge detection
bool         btnLastState         = HIGH;
unsigned long btnDownMs           = 0;

// MJPEG streaming
#define PART_BOUNDARY "HomeSightStream"
static const char* _STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART     =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ==========================================
// HomeSight Web Dashboard (embedded HTML)
// ==========================================
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HomeSight — Indoor Navigation AI</title>
    <meta name="description" content="HomeSight real-time hazard detection dashboard for visually impaired users.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

        :root {
            --bg:       #080810;
            --surface:  #10101e;
            --card:     #14142a;
            --border:   #22224a;
            --accent:   #7c6aff;
            --p0:       #ff3a5c;   /* Head hazard — red */
            --p1:       #ff9500;   /* Floor obstacle — amber */
            --p2:       #00d4ff;   /* Dynamic — cyan */
            --safe:     #00e676;   /* Clear path — green */
            --text:     #e8e8f8;
            --muted:    #7070a0;
            --mono:     'JetBrains Mono', monospace;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Ambient glow background */
        body::before {
            content: '';
            position: fixed; inset: 0;
            background:
                radial-gradient(ellipse 80% 60% at 20% 30%, rgba(124,106,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 70%, rgba(255,58,92,0.04) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 16px;
            position: relative;
            z-index: 1;
        }

        /* ── Header ── */
        .header {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px 0 16px;
            gap: 8px;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .logo-icon {
            width: 36px; height: 36px;
            background: linear-gradient(135deg, var(--accent), #ff6bb5);
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2em;
        }
        .logo h1 {
            font-size: 1.5em;
            font-weight: 700;
            background: linear-gradient(135deg, #a09fff, #ff6bb5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .tagline { color: var(--muted); font-size: 0.78em; letter-spacing: 0.5px; }

        /* ── Status badges ── */
        .status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 10px;
        }
        .badge {
            display: inline-flex; align-items: center; gap: 5px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.68em;
            font-weight: 600;
            letter-spacing: 0.3px;
            border: 1px solid;
        }
        .badge.ok     { background: rgba(0,230,118,0.08); border-color: rgba(0,230,118,0.25); color: var(--safe); }
        .badge.err    { background: rgba(255,58,92,0.08);  border-color: rgba(255,58,92,0.25);  color: var(--p0); }
        .badge.mode   { background: rgba(124,106,255,0.1); border-color: rgba(124,106,255,0.3); color: var(--accent); }
        .badge.warn   { background: rgba(255,149,0,0.08);  border-color: rgba(255,149,0,0.25);  color: var(--p1); }
        .dot {
            width: 6px; height: 6px;
            border-radius: 50%;
        }
        .dot.pulse { animation: dotPulse 1.8s ease-in-out infinite; }
        @keyframes dotPulse {
            0%,100% { opacity:1; transform:scale(1); }
            50%      { opacity:0.4; transform:scale(0.75); }
        }
        .dot.green { background: var(--safe); }
        .dot.red   { background: var(--p0); }
        .dot.blue  { background: var(--accent); }
        .dot.amber { background: var(--p1); }

        /* ── Priority Alert Banner ── */
        .priority-banner {
            display: none;
            align-items: center;
            gap: 12px;
            padding: 14px 18px;
            border-radius: 14px;
            margin: 14px 0;
            border: 1px solid;
            animation: bannerIn 0.25s cubic-bezier(0.34,1.56,0.64,1);
        }
        @keyframes bannerIn {
            from { opacity:0; transform:translateY(-8px) scale(0.97); }
            to   { opacity:1; transform:translateY(0) scale(1); }
        }
        .priority-banner.p0 {
            background: rgba(255,58,92,0.10);
            border-color: rgba(255,58,92,0.35);
        }
        .priority-banner.p1 {
            background: rgba(255,149,0,0.10);
            border-color: rgba(255,149,0,0.35);
        }
        .priority-banner.p2 {
            background: rgba(0,212,255,0.10);
            border-color: rgba(0,212,255,0.35);
        }
        .banner-icon { font-size: 1.6em; }
        .banner-body { flex: 1; }
        .banner-label {
            font-size: 0.65em;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            opacity: 0.8;
        }
        .banner-prompt {
            font-family: var(--mono);
            font-size: 1.05em;
            font-weight: 500;
            margin-top: 2px;
        }
        .p0 .banner-label  { color: var(--p0); }
        .p0 .banner-prompt { color: var(--p0); }
        .p1 .banner-label  { color: var(--p1); }
        .p1 .banner-prompt { color: var(--p1); }
        .p2 .banner-label  { color: var(--p2); }
        .p2 .banner-prompt { color: var(--p2); }

        /* ── Camera ── */
        .stream-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            margin: 14px 0;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        .stream-card.alert-glow {
            border-color: var(--p0);
            box-shadow: 0 0 30px rgba(255,58,92,0.15);
        }
        .stream-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            background: rgba(255,255,255,0.02);
            border-bottom: 1px solid var(--border);
            font-size: 0.73em;
            color: var(--muted);
        }
        .live-chip {
            display: flex; align-items: center; gap: 4px;
            color: var(--p0); font-weight: 700; font-size: 0.85em;
        }
        .stream-img {
            width: 100%; display: block;
            background: #000; min-height: 240px;
        }

        /* ── Controls ── */
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 14px 0;
        }
        .btn {
            padding: 13px 16px;
            border: none; border-radius: 12px;
            font-size: 0.88em; font-weight: 600;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
            display: flex; align-items: center; justify-content: center; gap: 6px;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:active { transform: translateY(0); }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        .btn-describe {
            background: linear-gradient(135deg, var(--accent), #a855f7);
            color: #fff;
            box-shadow: 0 4px 20px rgba(124,106,255,0.3);
        }
        .btn-describe:hover { box-shadow: 0 8px 28px rgba(124,106,255,0.45); }
        .btn-search {
            background: linear-gradient(135deg, #0d8a9e, #00d4ff);
            color: #fff;
            box-shadow: 0 4px 20px rgba(0,212,255,0.2);
        }
        .btn-search:hover { box-shadow: 0 8px 28px rgba(0,212,255,0.35); }

        /* ── Stats ── */
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin: 14px 0;
        }
        .stat {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 10px;
            text-align: center;
        }
        .stat-val {
            font-size: 1.35em;
            font-weight: 700;
            font-family: var(--mono);
        }
        .stat-val.accent  { color: var(--accent); }
        .stat-val.safe    { color: var(--safe); }
        .stat-val.danger  { color: var(--p0); }
        .stat-val.amber   { color: var(--p1); }
        .stat-lbl {
            font-size: 0.6em;
            color: var(--muted);
            margin-top: 3px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ── Alert Log ── */
        .log-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px;
            margin: 14px 0;
        }
        .log-card h3 {
            font-size: 0.75em;
            color: var(--muted);
            margin-bottom: 12px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .log-list {
            max-height: 280px;
            overflow-y: auto;
            scrollbar-width: thin;
        }
        .log-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 5px;
            font-size: 0.75em;
            animation: slideIn 0.25s ease;
        }
        @keyframes slideIn {
            from { opacity:0; transform:translateX(-6px); }
            to   { opacity:1; transform:translateX(0); }
        }
        .log-item.p0   { background:rgba(255,58,92,0.07);  border:1px solid rgba(255,58,92,0.15); }
        .log-item.p1   { background:rgba(255,149,0,0.07); border:1px solid rgba(255,149,0,0.15); }
        .log-item.p2   { background:rgba(0,212,255,0.07); border:1px solid rgba(0,212,255,0.15); }
        .log-item.info { background:rgba(124,106,255,0.06); border:1px solid rgba(124,106,255,0.15); }
        .log-tag {
            font-family: var(--mono);
            font-size: 0.7em;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 8px;
        }
        .p0 .log-tag   { background: rgba(255,58,92,0.2);  color: var(--p0); }
        .p1 .log-tag   { background: rgba(255,149,0,0.2); color: var(--p1); }
        .p2 .log-tag   { background: rgba(0,212,255,0.2); color: var(--p2); }
        .info .log-tag { background: rgba(124,106,255,0.2); color: var(--accent); }
        .log-time { color: var(--muted); font-size: 0.85em; white-space: nowrap; margin-left: 8px; }
        .no-logs { text-align:center; color:var(--muted); font-size:0.8em; padding:24px; }

        /* ── Audio ── */
        #audioSection { margin: 14px 0; display: none; }
        #audioSection audio { width:100%; border-radius: 8px; }

        /* ── Footer ── */
        .footer {
            text-align: center;
            padding: 18px;
            color: var(--muted);
            font-size: 0.62em;
            letter-spacing: 0.5px;
        }

        @media (max-width: 560px) {
            .logo h1 { font-size: 1.2em; }
            .stats { grid-template-columns: repeat(2,1fr); }
            .controls { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <div class="logo">
            <div class="logo-icon">👓</div>
            <h1>HomeSight</h1>
        </div>
        <div class="tagline">Indoor Navigation AI — Offline Edge Inference</div>
        <div class="status-row">
            <div class="badge ok" id="wifiBadge"><div class="dot green pulse"></div>WiFi</div>
            <div class="badge err" id="serverBadge"><div class="dot red pulse"></div>Server</div>
            <div class="badge mode" id="modeBadge"><div class="dot blue pulse"></div>Detecting</div>
            <div class="badge ok" id="priorityBadge"><div class="dot green"></div>CLEAR</div>
        </div>
    </div>

    <!-- Priority Alert Banner -->
    <div class="priority-banner" id="priorityBanner">
        <div class="banner-icon" id="bannerIcon">⚠️</div>
        <div class="banner-body">
            <div class="banner-label" id="bannerLabel">P0 — HEAD HAZARD</div>
            <div class="banner-prompt" id="bannerPrompt">Watch head — cabinet</div>
        </div>
    </div>

    <!-- Camera Stream -->
    <div class="stream-card" id="streamCard">
        <div class="stream-header">
            <span>📷 OV3660 — 110° FOV</span>
            <div class="live-chip"><div class="dot red pulse"></div>LIVE</div>
        </div>
        <img class="stream-img" id="stream" src="/stream" alt="Camera feed" loading="lazy">
    </div>

    <!-- Controls -->
    <div class="controls">
        <button class="btn btn-describe" id="btnDescribe" onclick="describeScene()">
            🎤 Describe Scene
        </button>
        <button class="btn btn-search" id="btnSearch" onclick="openSearch()">
            🔍 Find Object
        </button>
    </div>

    <!-- Audio Player -->
    <div id="audioSection">
        <audio id="audioPlayer" controls autoplay></audio>
    </div>

    <!-- Stats -->
    <div class="stats">
        <div class="stat">
            <div class="stat-val danger" id="alertCount">0</div>
            <div class="stat-lbl">Alerts</div>
        </div>
        <div class="stat">
            <div class="stat-val safe" id="uptime">0s</div>
            <div class="stat-lbl">Uptime</div>
        </div>
        <div class="stat">
            <div class="stat-val accent" id="latencyVal">--</div>
            <div class="stat-lbl">Last ms</div>
        </div>
        <div class="stat">
            <div class="stat-val" id="serverStatus">--</div>
            <div class="stat-lbl">Server</div>
        </div>
    </div>

    <!-- Alert Log -->
    <div class="log-card">
        <h3>📋 Hazard Log</h3>
        <div class="log-list" id="logList">
            <div class="no-logs">Monitoring... path is clear.</div>
        </div>
    </div>

    <div class="footer">
        HomeSight MVP v3.0 &nbsp;·&nbsp; ESP32-S3 + YOLOv8n + pyttsx3 &nbsp;·&nbsp; Offline Edge Inference
    </div>
</div>

<script>
    let lastPriority = 'CLEAR';

    const PRIORITY_CONFIG = {
        P0:    { cls:'p0',  icon:'🚨', label:'P0 — HEAD HAZARD',   badgeClass:'err',  badgeText:'P0 HEAD' },
        P1:    { cls:'p1',  icon:'⚠️', label:'P1 — FLOOR OBSTACLE', badgeClass:'warn', badgeText:'P1 FLOOR' },
        P2:    { cls:'p2',  icon:'🔵', label:'P2 — DYNAMIC OBJECT', badgeClass:'mode', badgeText:'P2 DYNAMIC' },
        CLEAR: { cls:'',   icon:'✅', label:'',                     badgeClass:'ok',   badgeText:'CLEAR' },
    };

    function updateStatus() {
        fetch('/status').then(r => r.json()).then(data => {
            // Mode badge
            const mb = document.getElementById('modeBadge');
            const modeLabels = {
                detecting: 'Detecting', describing: 'Describing…',
                locating:  'Locating…', standby: 'Standby', idle: 'Idle'
            };
            mb.innerHTML = `<div class="dot blue pulse"></div>${modeLabels[data.mode] || data.mode}`;

            // Server badge
            const sb = document.getElementById('serverBadge');
            if (data.serverOk) {
                sb.className = 'badge ok';
                sb.innerHTML = '<div class="dot green pulse"></div>Server';
                document.getElementById('serverStatus').textContent = '✅';
            } else {
                sb.className = 'badge err';
                sb.innerHTML = '<div class="dot red pulse"></div>Server';
                document.getElementById('serverStatus').textContent = '❌';
            }

            // Priority banner
            const pri = data.lastPriority || 'CLEAR';
            if (pri !== 'CLEAR' && pri !== lastPriority) {
                showPriorityBanner(pri, data.lastMessage);
            } else if (pri === 'CLEAR') {
                hidePriorityBanner();
            }
            lastPriority = pri;

            // Priority badge
            const cfg = PRIORITY_CONFIG[pri] || PRIORITY_CONFIG['CLEAR'];
            const pb  = document.getElementById('priorityBadge');
            pb.className = 'badge ' + cfg.badgeClass;
            pb.innerHTML = `<div class="dot ${pri === 'P0' ? 'red' : pri === 'P1' ? 'amber' : 'green'} pulse"></div>${cfg.badgeText}`;

            // Uptime
            const s = data.uptime % 60;
            const m = Math.floor(data.uptime / 60);
            document.getElementById('uptime').textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
            document.getElementById('alertCount').textContent = data.alertCount;
        }).catch(() => {});
    }

    function showPriorityBanner(priority, prompt) {
        const cfg = PRIORITY_CONFIG[priority];
        if (!cfg || !cfg.label) return;
        const banner = document.getElementById('priorityBanner');
        banner.className = 'priority-banner ' + cfg.cls;
        document.getElementById('bannerIcon').textContent   = cfg.icon;
        document.getElementById('bannerLabel').textContent  = cfg.label;
        document.getElementById('bannerPrompt').textContent = prompt || '';
        banner.style.display = 'flex';

        // Vibrate stream card border
        const sc = document.getElementById('streamCard');
        sc.classList.add('alert-glow');

        clearTimeout(banner._timer);
        banner._timer = setTimeout(() => hidePriorityBanner(), 6000);
    }

    function hidePriorityBanner() {
        document.getElementById('priorityBanner').style.display = 'none';
        document.getElementById('streamCard').classList.remove('alert-glow');
    }

    function updateAlerts() {
        fetch('/alerts').then(r => r.json()).then(alerts => {
            const el = document.getElementById('logList');
            if (!alerts.length) {
                el.innerHTML = '<div class="no-logs">Monitoring... path is clear.</div>';
                return;
            }
            const priLabel = { P0:'P0', P1:'P1', P2:'P2', INFO:'INFO' };
            let html = '';
            for (let i = alerts.length - 1; i >= 0; i--) {
                const a = alerts[i];
                const priKey = a.priority || (a.danger ? 'P0' : 'INFO');
                const cls    = priKey.toLowerCase().replace('info', 'info');
                const m = Math.floor(a.t / 60), s = a.t % 60;
                const ts = m > 0 ? `${m}m ${s}s` : `${s}s`;
                html += `<div class="log-item ${cls}">`;
                html += `<span><span class="log-tag">${priLabel[priKey] || priKey}</span>${a.msg}</span>`;
                html += `<span class="log-time">${ts}</span>`;
                html += '</div>';
            }
            el.innerHTML = html;
        }).catch(() => {});
    }

    function describeScene() {
        const btn = document.getElementById('btnDescribe');
        btn.disabled = true;
        btn.innerHTML = '⏳ Analysing…';
        fetch('/trigger-describe', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                btn.disabled = false;
                btn.innerHTML = '🎤 Describe Scene';
                if (data.audio_base64) {
                    const audio = document.getElementById('audioPlayer');
                    audio.src = 'data:audio/wav;base64,' + data.audio_base64;
                    document.getElementById('audioSection').style.display = 'block';
                    audio.play();
                }
                updateAlerts();
            })
            .catch(() => {
                btn.disabled = false;
                btn.innerHTML = '🎤 Describe Scene';
            });
    }

    function openSearch() {
        const query = prompt('🔍 What are you looking for?\n(e.g. "keys", "water cup", "phone")');
        if (!query || !query.trim()) return;
        const btn = document.getElementById('btnSearch');
        btn.disabled = true;
        btn.innerHTML = '⏳ Searching…';
        fetch('/trigger-search?q=' + encodeURIComponent(query.trim()), { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                btn.disabled = false;
                btn.innerHTML = '🔍 Find Object';
                if (data.audio_base64) {
                    const audio = document.getElementById('audioPlayer');
                    audio.src = 'data:audio/wav;base64,' + data.audio_base64;
                    document.getElementById('audioSection').style.display = 'block';
                    audio.play();
                }
                updateAlerts();
            })
            .catch(() => {
                btn.disabled = false;
                btn.innerHTML = '🔍 Find Object';
            });
    }

    // Poll status every 800 ms, alerts every 2 s
    setInterval(updateStatus, 800);
    setInterval(updateAlerts,  2000);
    updateStatus();
    updateAlerts();
</script>
</body>
</html>
)rawliteral";

// ==========================================
// Web Server Handlers
// ==========================================
void handleRoot()   { server.send(200, "text/html", INDEX_HTML); }
void handleStatus() { server.send(200, "application/json", Logic.getStatusJSON(currentMode, serverReady)); }
void handleAlerts() { server.send(200, "application/json", Logic.getAlertHistoryJSON()); }

void handleStream() {
    WiFiClient client = server.client();
    String resp  = "HTTP/1.1 200 OK\r\n";
    resp += "Content-Type: ";
    resp += _STREAM_CONTENT_TYPE;
    resp += "\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    client.print(resp);

    while (client.connected()) {
        camera_fb_t* fb = Camera.captureFrame();
        if (!fb) { delay(100); continue; }
        char partBuf[64];
        snprintf(partBuf, sizeof(partBuf), _STREAM_PART, fb->len);
        client.print(_STREAM_BOUNDARY);
        client.print(partBuf);
        client.write(fb->buf, fb->len);
        Camera.releaseFrame(fb);
        delay(80);  // ~12 FPS
    }
}

void handleTriggerDescribe() {
    if (!cameraReady) { server.send(500, "application/json", "{\"error\":\"Camera not ready\"}"); return; }

    currentMode = MODE_DESCRIBING;
    camera_fb_t* fb = Camera.captureFrame();
    if (!fb)        { server.send(500, "application/json", "{\"error\":\"Capture failed\"}"); currentMode = MODE_DETECTING; return; }

    ServerResponse resp = Cloud.describeImage(fb);
    Camera.releaseFrame(fb);
    currentMode = MODE_DETECTING;

    if (!resp.success) { server.send(500, "application/json", "{\"error\":\"Server unreachable\"}"); return; }

    Logic.processDescribeResult(resp);

    String json = "{";
    json += "\"message\":\"" + resp.message + "\"";
    if (resp.audioData && resp.audioLength > 0) {
        size_t b64Len = ((resp.audioLength + 2) / 3) * 4 + 1;
        char* b64     = (char*)malloc(b64Len);
        if (b64) {
            size_t actual = 0;
            mbedtls_base64_encode((uint8_t*)b64, b64Len, &actual, resp.audioData, resp.audioLength);
            b64[actual] = '\0';
            json += ",\"audio_base64\":\"" + String(b64) + "\"";
            free(b64);
        }
        free(resp.audioData);
    }
    json += ",\"elapsed\":" + String(resp.elapsed);
    json += "}";
    server.send(200, "application/json", json);
}

void handleTriggerSearch() {
    String query = server.arg("q");
    if (query.length() == 0) { server.send(400, "application/json", "{\"error\":\"No query\"}"); return; }
    if (!cameraReady)        { server.send(500, "application/json", "{\"error\":\"Camera not ready\"}"); return; }

    currentMode = MODE_LOCATING;
    Audio.playConfirmChirp();   // confirm mode switch

    camera_fb_t* fb = Camera.captureFrame();
    if (!fb) { currentMode = MODE_DETECTING; server.send(500, "application/json", "{\"error\":\"Capture failed\"}"); return; }

    ServerResponse resp = Cloud.searchObject(fb, query);
    Camera.releaseFrame(fb);
    currentMode = MODE_DETECTING;

    Logic.processSearchResult(resp);

    String json = "{";
    json += "\"found\":"    + String(resp.hasAlert ? "true" : "false") + ",";
    json += "\"message\":\"" + resp.message + "\"";
    if (resp.audioData && resp.audioLength > 0) {
        size_t b64Len = ((resp.audioLength + 2) / 3) * 4 + 1;
        char* b64     = (char*)malloc(b64Len);
        if (b64) {
            size_t actual = 0;
            mbedtls_base64_encode((uint8_t*)b64, b64Len, &actual, resp.audioData, resp.audioLength);
            b64[actual] = '\0';
            json += ",\"audio_base64\":\"" + String(b64) + "\"";
            free(b64);
        }
        free(resp.audioData);
    }
    if (resp.clickAudioData) free(resp.clickAudioData);
    json += "}";
    server.send(200, "application/json", json);
}

// ==========================================
// Setup
// ==========================================
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println();
    Serial.println("╔══════════════════════════════════════════════╗");
    Serial.println("║   HomeSight — Indoor Navigation AI v3.0      ║");
    Serial.println("║   Offline Edge Inference · No Cloud Required  ║");
    Serial.println("╚══════════════════════════════════════════════╝");
    Serial.println();

    // Button — uses internal pull-up; active LOW
    pinMode(PIN_ACTION_BTN, INPUT_PULLUP);

    // Speaker (open-ear / bone-conduction)
    speakerReady = Audio.begin();
    if (speakerReady) Audio.playConfirmChirp();

    // Camera
    cameraReady = Camera.begin();
    Serial.println(cameraReady ? "[OK] Camera" : "[FAIL] Camera");

    // WiFi
    if (Cloud.connectWiFi()) {
        Serial.println("[OK] WiFi connected");
    } else {
        Serial.println("[!!] WiFi failed — starting AP mode");
        Cloud.startAP();
    }

    // Check local Python server
    serverReady = Cloud.checkServer();
    if (!serverReady) {
        Serial.println();
        Serial.println("⚠  HomeSight Python server not found.");
        Serial.println("   1.  cd python_server");
        Serial.println("   2.  python models/download_model.py");
        Serial.println("   3.  pip install -r requirements.txt");
        Serial.println("   4.  python server.py");
        Serial.printf( "   5.  Set PYTHON_SERVER_IP in Config.h to your computer's IP\n");
        Serial.println("   (Retrying every 10 seconds)");
    }

    String ip = Cloud.getIP();

    Serial.println();
    Serial.println("╔══════════════════════════════════════════════╗");
    Serial.printf( "║  URL:    http://%-28s ║\n", ip.c_str());
    Serial.printf( "║  Server: %-35s ║\n", serverReady ? "✅ Local YOLOv8n server ready" : "❌ Not found");
    Serial.println("╚══════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("Controls:");
    Serial.println("  Single press  → Describe scene");
    Serial.println("  Double tap    → Object locator mode");
    Serial.println();

    // Web server routes
    server.on("/",                HTTP_GET,  handleRoot);
    server.on("/stream",          HTTP_GET,  handleStream);
    server.on("/status",          HTTP_GET,  handleStatus);
    server.on("/alerts",          HTTP_GET,  handleAlerts);
    server.on("/trigger-describe",HTTP_POST, handleTriggerDescribe);
    server.on("/trigger-search",  HTTP_POST, handleTriggerSearch);
    server.begin();
}

// ==========================================
// Main Loop
// ==========================================
void loop() {
    server.handleClient();

    unsigned long now = millis();

    // ─── Battery Monitor ────────────────────────────────────────────────────
    if (now - lastBatteryCheck > BATTERY_CHECK_INTERVAL_MS) {
        lastBatteryCheck = now;
        int battPct = Battery.getPercent();
        if (battPct > 0 && battPct <= BATTERY_LOW_PERCENT) {
            if (now - lastBatteryAlertMs > 60000) {   // max once per minute
                lastBatteryAlertMs = now;
                Serial.printf("[BAT] Low battery: %d%%\n", battPct);
                Audio.playEarconP0();  // Critical earcon for battery too
                Logic.addAlert("Battery low — " + String(battPct) + "%", PRIORITY_P0);
            }
        }
    }

    // ─── Button Edge Detection + Double-tap ─────────────────────────────────
    bool btnCurrent = digitalRead(PIN_ACTION_BTN);

    if (btnLastState == HIGH && btnCurrent == LOW) {
        // Button just pressed (falling edge)
        if (now - btnDownMs > BUTTON_DEBOUNCE_MS) {
            btnDownMs = now;
            bool isDoubleTap = Logic.doubleTap.update(now);

            if (isDoubleTap) {
                // ── Double-tap: Object Locator ──────────────────────────────
                Serial.println("\n🔍 Double-tap — Object Locator mode");
                currentMode = MODE_LOCATING;
                Audio.playConfirmChirp();
                // In a real device, microphone STT would capture the query here.
                // For now, a default query is used (replace with MicMod + STT).
                String query = "keys";
                Serial.println("[LOCATOR] Query: '" + query + "'");

                if (cameraReady && serverReady) {
                    camera_fb_t* fb = Camera.captureFrame();
                    if (fb) {
                        ServerResponse resp = Cloud.searchObject(fb, query);
                        Camera.releaseFrame(fb);
                        Logic.processSearchResult(resp);
                        if (resp.audioData) free(resp.audioData);
                        if (resp.clickAudioData) free(resp.clickAudioData);
                    }
                }
                currentMode = MODE_DETECTING;

            } else {
                // ── First press — wait to see if double-tap follows ─────────
                // (Handled below when window expires)
            }
        }
    }

    // When double-tap window expires with only one press → single press action
    if (Logic.doubleTap.windowExpired(now) && Logic.doubleTap.waitingSecond) {
        Logic.doubleTap.waitingSecond = false;
        // ── Single press: Scene Description ──────────────────────────────
        Serial.println("\n🎤 Single press — scene description");
        currentMode = MODE_DESCRIBING;
        if (speakerReady) Audio.playTone(660, 80, 0.35f);

        if (cameraReady && serverReady) {
            camera_fb_t* fb = Camera.captureFrame();
            if (fb) {
                ServerResponse resp = Cloud.describeImage(fb);
                Camera.releaseFrame(fb);
                if (resp.success) {
                    Logic.processDescribeResult(resp);
                    Serial.println("📝 " + resp.message);
                }
                if (resp.audioData) free(resp.audioData);
            }
        }
        currentMode = MODE_DETECTING;
    }

    btnLastState = btnCurrent;

    // ─── Auto Hazard Detection ───────────────────────────────────────────────
    // Runs every AUTO_DETECT_INTERVAL_MS (1500 ms with local YOLO server)
    if (cameraReady && serverReady && currentMode == MODE_DETECTING
        && (now - lastAutoDetectTime > AUTO_DETECT_INTERVAL_MS)) {

        lastAutoDetectTime = now;

        camera_fb_t* fb = Camera.captureFrame();
        if (fb) {
            ServerResponse resp = Cloud.analyzeImage(fb);
            Camera.releaseFrame(fb);

            if (resp.success) {
                Logic.processHazardResult(resp);
            }
            if (resp.audioData)      free(resp.audioData);
            if (resp.clickAudioData) free(resp.clickAudioData);
        }
    }

    // ─── Retry Server Connection (every 10 s) ────────────────────────────────
    if (!serverReady && (now - lastServerCheck > 10000)) {
        lastServerCheck = now;
        serverReady = Cloud.checkServer();
        if (serverReady) {
            Serial.println("✅ HomeSight server connected!");
            Logic.addAlert("Server connected", PRIORITY_NORMAL);
            Audio.playConfirmChirp();
        }
    }

    delay(1);
}
