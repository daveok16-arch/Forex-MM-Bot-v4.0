import os
import sys
import threading
import asyncio
import time
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)
scanner_running = False
scanner_thread = None
startup_logs = []

def log(msg):
    startup_logs.append(msg)
    print(f"[APP] {msg}")

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "scanner_running": scanner_running,
        "startup_logs": startup_logs[-10:],
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
        "startup_logs": startup_logs[-15:],
        "timestamp": datetime.utcnow().isoformat()
    })

def run_scanner():
    global scanner_running
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log(f"Base dir: {base_dir}")
        config_path = os.path.join(base_dir, "config", "mm_config.yaml")
        log(f"Config: {config_path}, exists: {os.path.exists(config_path)}")
        sys.path.insert(0, base_dir)
        log("Importing scanner...")
        from scanner.v6_scanner import V6Scanner
        log("Scanner imported")
        log("Creating V6Scanner...")
        scanner = V6Scanner(config_path)
        log("V6Scanner created")
        scanner_running = True
        log("Starting run loop...")
        asyncio.run(scanner.run(300))
    except Exception as e:
        import traceback
        log(f"CRASH: {e}")
        log(traceback.format_exc())
        scanner_running = False

log("Starting scanner thread...")
scanner_thread = threading.Thread(target=run_scanner, daemon=True)
scanner_thread.start()

for i in range(20):
    time.sleep(1)
    if scanner_running:
        log(f"CONFIRMED: Scanner running after {i}s")
        break
    if not scanner_thread.is_alive():
        log(f"Thread died after {i}s")
        break
else:
    log(f"Thread alive after 20s, scanner_running={scanner_running}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
