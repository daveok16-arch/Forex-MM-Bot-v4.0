# app.py — Flask wrapper for Render free Web Service
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

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "scanner_running": scanner_running,
        "last_scan": str(last_scan_time) if last_scan_time else None,
        "scan_count": scan_count,
        "scanner_error": scanner_error,
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
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "error": scanner_error
    })

def run_scanner():
    global scanner_running, last_scan_time, scan_count, scanner_error
    
    # Use relative path from project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "mm_config.yaml")
    
    # Ensure config exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if not os.path.exists(config_path):
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
    
    sys.path.insert(0, base_dir)
    
    try:
        print(f"[App] Loading scanner with config: {config_path}")
        from scanner.v6_scanner import V6Scanner
        
        scanner_running = True
        scanner_error = None
        interval = float(os.environ.get("SCAN_INTERVAL", "300"))
        
        print(f"[App] Initializing V6Scanner...")
        scanner = V6Scanner(config_path)
        print(f"[App] V6Scanner initialized. Starting run loop...")
        
        asyncio.run(scanner.run(interval))
        
    except Exception as e:
        scanner_error = str(e)
        print(f"[App] Scanner crashed: {e}")
        traceback.print_exc()
        scanner_running = False

# AUTO-START SCANNER
print("[App] Starting background scanner thread...")
scanner_thread = threading.Thread(target=run_scanner, daemon=True)
scanner_thread.start()
time.sleep(3)
print(f"[App] Scanner thread started. Running: {scanner_running}, Error: {scanner_error}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
