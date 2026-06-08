# app.py — With forced import completion and granular logging
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
    ts = datetime.utcnow().strftime('%H:%M:%S')
    entry = f"{ts} {msg}"
    startup_logs.append(entry)
    print(f"[APP] {msg}")

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "scanner_running": scanner_running,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "last_error": last_error,
        "startup_logs": startup_logs[-20:],
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
        "startup_logs": startup_logs[-25:],
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
        
        # Import step by step
        log("Importing yaml...")
        import yaml
        log("yaml OK")
        
        log("Importing numpy...")
        import numpy as np
        log("numpy OK")
        
        log("Importing onnxruntime...")
        import onnxruntime as ort
        log("onnxruntime OK")
        
        log("Importing requests...")
        import requests
        log("requests OK")
        
        log("Importing yfinance...")
        import yfinance as yf
        log("yfinance OK")
        
        log("Importing V6Scanner...")
        from scanner.v6_scanner import V6Scanner
        log("V6Scanner imported")
        
        log("Creating V6Scanner instance...")
        scanner = V6Scanner(config_path)
        log("V6Scanner created")
        
        # Test Telegram
        log("Testing Telegram...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scanner.tg._send("🔄 Render test - " + datetime.utcnow().isoformat()))
            loop.close()
            log(f"Telegram test result: {result}")
        except Exception as e:
            log(f"Telegram test failed: {e}")
        
        scanner_running = True
        last_error = None
        log("Starting run loop...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(scanner.run(300))
        loop.close()
        
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

# Monitor and restart
def monitor_thread():
    global scanner_thread
    while True:
        time.sleep(30)
        if scanner_thread and not scanner_thread.is_alive():
            log("Thread died! Restarting...")
            scanner_running = False
            start_scanner()

monitor = threading.Thread(target=monitor_thread, daemon=True)
monitor.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
