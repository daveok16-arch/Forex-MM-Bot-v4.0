# app.py — DIAGNOSTIC VERSION for Render
import os
import sys
import threading
import asyncio
import time
import traceback
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

scanner_running = False
last_scan_time = None
scan_count = 0
scanner_thread = None
scanner_error = None
init_logs = []

def log(msg):
    init_logs.append(msg)
    print(msg)

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "scanner_running": scanner_running,
        "last_scan": str(last_scan_time) if last_scan_time else None,
        "scan_count": scan_count,
        "scanner_error": scanner_error,
        "init_logs": init_logs,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health")
def detailed_health():
    return jsonify({
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/status")
def scanner_status():
    return jsonify({
        "scanner_running": scanner_running,
        "last_scan": str(last_scan_time) if last_scan_time else None,
        "scan_count": scan_count,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "error": scanner_error,
        "init_logs": init_logs
    })

def run_scanner():
    global scanner_running, last_scan_time, scan_count, scanner_error
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log(f"[1] Base dir: {base_dir}")
        
        config_path = os.path.join(base_dir, "config", "mm_config.yaml")
        log(f"[2] Config path: {config_path}")
        log(f"[3] Config exists: {os.path.exists(config_path)}")
        
        if not os.path.exists(config_path):
            log("[4] Creating default config...")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write("""
paths:
  drive_base: ./data
  models_dir: data/models
microstructure:
  window: 20
  volume_threshold: 1.5
telegram:
  bot_token: ${TELEGRAM_BOT_TOKEN}
  chat_id: ${TELEGRAM_CHAT_ID}
""")
        
        # Read config to verify
        with open(config_path, "r") as f:
            cfg_content = f.read()
        log(f"[5] Config content: {cfg_content[:200]}...")
        
        sys.path.insert(0, base_dir)
        log("[6] sys.path updated")
        
        log("[7] Importing V6Scanner...")
        from scanner.v6_scanner import V6Scanner
        log("[8] V6Scanner imported successfully")
        
        log("[9] Creating V6Scanner instance...")
        scanner = V6Scanner(config_path)
        log("[10] V6Scanner created successfully")
        log(f"[11] Ensemble loaded: {scanner.ensemble}")
        
        scanner_running = True
        scanner_error = None
        interval = float(os.environ.get("SCAN_INTERVAL", "300"))
        log(f"[12] Starting run loop with interval: {interval}")
        
        asyncio.run(scanner.run(interval))
        
    except Exception as e:
        scanner_error = str(e)
        log(f"[ERROR] Scanner crashed: {e}")
        traceback_str = traceback.format_exc()
        log(f"[TRACE] {traceback_str}")
        scanner_running = False

# AUTO-START
log("[0] Starting background scanner thread...")
scanner_thread = threading.Thread(target=run_scanner, daemon=True)
scanner_thread.start()
time.sleep(5)
log(f"[END] Thread alive: {scanner_thread.is_alive()}, Running: {scanner_running}, Error: {scanner_error}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
