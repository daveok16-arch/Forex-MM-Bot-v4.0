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
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health")
def detailed_health():
    return jsonify({"status": "running", "timestamp": datetime.utcnow().isoformat()})

@app.route("/status")
def scanner_status():
    return jsonify({
        "scanner_running": scanner_running,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "timestamp": datetime.utcnow().isoformat()
    })

def run_scanner():
    global scanner_running
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "mm_config.yaml")
    sys.path.insert(0, base_dir)
    from scanner.v6_scanner import V6Scanner
    scanner = V6Scanner(config_path)
    scanner_running = True
    asyncio.run(scanner.run(300))

scanner_thread = threading.Thread(target=run_scanner, daemon=True)
scanner_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
