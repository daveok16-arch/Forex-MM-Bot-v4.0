# app.py — Robust Flask wrapper with thread restart
import os
import sys
import threading
import asyncio
import time
import traceback
from datetime import datetime
from flask import Flask, jsonify

# Force CPU-only ONNX
os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_DISABLE_GPU"] = "1"

app = Flask(__name__)

scanner_running = False
scanner_thread = None
startup_logs = []
last_error = None

def log(msg):
    startup_logs.append(f"{datetime.utcnow().strftime('%H:%M:%S')} {msg}")
    print(f"[APP] {msg}")

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "scanner_running": scanner_running,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "last_error": last_error,
        "startup_logs": startup_logs[-15:],
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
        "last_error": last_error,
        "startup_logs": startup_logs[-20:],
        "timestamp": datetime.utcnow().isoformat()
    })

def run_scanner():
    global scanner_running, last_error
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
        
        # Test Telegram before starting loop
        log("Testing Telegram...")
        try:
            asyncio.run(scanner.tg._send("🔄 Render bot restarted - test message"))
            log("Telegram test OK")
        except Exception as e:
            log(f"Telegram test failed: {e}")
        
        scanner_running = True
        last_error = None
        log("Starting run loop...")
        asyncio.run(scanner.run(300))
        
    except Exception as e:
        last_error = str(e)
        log(f"CRASH: {e}")
        log(traceback.format_exc())
        scanner_running = False

def start_scanner():
    global scanner_thread
    log("Starting scanner thread...")
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()

# Initial start
start_scanner()

# Monitor and restart if thread dies
def monitor_thread():
    global scanner_thread
    while True:
        time.sleep(60)
        if scanner_thread and not scanner_thread.is_alive():
            log("Thread died! Restarting...")
            scanner_running = False
            start_scanner()

monitor = threading.Thread(target=monitor_thread, daemon=True)
monitor.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
