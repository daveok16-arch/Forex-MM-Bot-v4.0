# app.py — Flask wrapper for Render free Web Service
import os
import sys
import threading
import asyncio
import time
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

scanner_running = False
last_scan_time = None
scan_count = 0
scanner_thread = None

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "scanner_running": scanner_running,
        "last_scan": str(last_scan_time) if last_scan_time else None,
        "scan_count": scan_count,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health")
def detailed_health():
    return jsonify({
        "status": "running",
        "uptime": "active",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/status")
def scanner_status():
    return jsonify({
        "scanner_running": scanner_running,
        "last_scan": str(last_scan_time) if last_scan_time else None,
        "scan_count": scan_count,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False
    })

def run_scanner():
    global scanner_running, last_scan_time, scan_count
    
    config_path = os.environ.get(
        "CONFIG_PATH", 
        "/opt/render/project/src/config/mm_config.yaml"
    )
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("""
paths:
  drive_base: /opt/render/project/src/data
  models_dir: models
microstructure:
  window: 20
  volume_threshold: 1.5
telegram:
  bot_token: ${TELEGRAM_BOT_TOKEN}
  chat_id: ${TELEGRAM_CHAT_ID}
""")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from scanner.v6_scanner import V6Scanner
        scanner_running = True
        interval = float(os.environ.get("SCAN_INTERVAL", "300"))
        scanner = V6Scanner(config_path)
        asyncio.run(scanner.run(interval))
    except Exception as e:
        print(f"[App] Scanner crashed: {e}")
        import traceback
        traceback.print_exc()
        scanner_running = False

# 🔴 AUTO-START SCANNER WHEN MODULE LOADS (Gunicorn imports this)
print("[App] Starting background scanner thread...")
scanner_thread = threading.Thread(target=run_scanner, daemon=True)
scanner_thread.start()
time.sleep(2)  # Give scanner time to initialize
print(f"[App] Scanner thread started. Running: {scanner_running}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
