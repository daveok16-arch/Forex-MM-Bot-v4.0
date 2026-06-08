# app.py — Flask wrapper for Render free Web Service
import os
import sys
import threading
import asyncio
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

scanner_running = False
last_scan_time = None
scan_count = 0

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

def run_scanner():
    global scanner_running, last_scan_time, scan_count
    
    config_path = os.environ.get(
        "CONFIG_PATH", 
        "/app/config/mm_config.yaml"
    )
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("""
paths:
  drive_base: /app/data
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
        scanner_running = False

@app.route("/start")
def start_scanner():
    global scanner_thread, scanner_running
    if not scanner_running:
        scanner_thread = threading.Thread(target=run_scanner, daemon=True)
        scanner_thread.start()
        return jsonify({"status": "scanner started"})
    return jsonify({"status": "already running"})

if __name__ == "__main__":
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
