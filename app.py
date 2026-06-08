# app.py — With configurable interval and better status
import os
import sys
import threading
import asyncio
import time
import traceback
import importlib
import logging
from datetime import datetime
from flask import Flask, jsonify

# Force CPU-only ONNX
os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_DISABLE_GPU"] = "1"

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

scanner_running = False
scanner_thread = None
startup_logs = []
last_error = None
restart_count = 0
monitor_checks = 0

def log(msg):
    ts = datetime.utcnow().strftime('%H:%M:%S')
    entry = f"{ts} {msg}"
    startup_logs.append(entry)
    logger.info(f"[APP] {msg}")
    sys.stdout.flush()

def safe_import(module_name):
    try:
        log(f"Importing {module_name}...")
        module = importlib.import_module(module_name)
        log(f"{module_name} OK")
        sys.stdout.flush()
        return module
    except Exception as e:
        log(f"{module_name} FAILED: {e}")
        log(traceback.format_exc())
        sys.stdout.flush()
        return None

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "scanner_running": scanner_running,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "last_error": last_error,
        "restart_count": restart_count,
        "monitor_checks": monitor_checks,
        "startup_logs": startup_logs[-30:],
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
        "restart_count": restart_count,
        "monitor_checks": monitor_checks,
        "startup_logs": startup_logs[-35:],
        "timestamp": datetime.utcnow().isoformat()
    })

def run_scanner():
    global scanner_running, last_error
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log(f"Base dir: {base_dir}")
        
        config_path = os.path.join(base_dir, "config", "mm_config.yaml")
        log(f"Config: {config_path}, exists: {os.path.exists(config_path)}")
        sys.stdout.flush()
        
        sys.path.insert(0, base_dir)
        
        # Import modules
        yaml = safe_import("yaml")
        if not yaml: return
        
        np = safe_import("numpy")
        if not np: return
        
        ort = safe_import("onnxruntime")
        if not ort: log("onnxruntime failed, continuing")
        
        requests = safe_import("requests")
        if not requests: return
        
        yf = safe_import("yfinance")
        if not yf: return
        
        log("Importing V6Scanner...")
        sys.stdout.flush()
        try:
            from scanner.v6_scanner import V6Scanner
            log("V6Scanner imported")
            sys.stdout.flush()
        except Exception as e:
            log(f"V6Scanner import FAILED: {e}")
            log(traceback.format_exc())
            sys.stdout.flush()
            return
        
        log("Creating V6Scanner...")
        sys.stdout.flush()
        try:
            scanner = V6Scanner(config_path)
            log("V6Scanner created")
            sys.stdout.flush()
        except Exception as e:
            log(f"V6Scanner creation FAILED: {e}")
            log(traceback.format_exc())
            sys.stdout.flush()
            return
        
        # Test Telegram
        log("Testing Telegram...")
        sys.stdout.flush()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scanner.tg._send("🔄 Render restart test - " + datetime.utcnow().isoformat()))
            loop.close()
            log(f"Telegram test: {result}")
            sys.stdout.flush()
        except Exception as e:
            log(f"Telegram test failed: {e}")
            sys.stdout.flush()
        
        scanner_running = True
        last_error = None
        interval = int(os.environ.get("SCAN_INTERVAL", "1800"))
        log(f"Starting run loop with interval: {interval}s")
        sys.stdout.flush()
        
        # Run with comprehensive error catching
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(scanner.run(interval))
        except SystemExit as e:
            log(f"SystemExit caught: {e}")
            sys.stdout.flush()
        except KeyboardInterrupt as e:
            log(f"KeyboardInterrupt caught: {e}")
            sys.stdout.flush()
        except Exception as e:
            log(f"Loop error: {e}")
            log(traceback.format_exc())
            sys.stdout.flush()
        finally:
            loop.close()
            log("Loop closed")
            sys.stdout.flush()
        
    except Exception as e:
        last_error = str(e)
        log(f"CRASH: {e}")
        log(traceback.format_exc())
        sys.stdout.flush()
        scanner_running = False

def start_scanner():
    global scanner_thread, restart_count
    restart_count += 1
    log(f"Starting scanner thread (restart #{restart_count})...")
    sys.stdout.flush()
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()

# Initial start
start_scanner()

# Monitor and restart
def monitor_thread():
    global scanner_thread, monitor_checks
    log("Monitor thread started")
    sys.stdout.flush()
    while True:
        time.sleep(10)
        monitor_checks += 1
        alive = scanner_thread.is_alive() if scanner_thread else False
        log(f"Monitor check #{monitor_checks}, thread alive: {alive}")
        sys.stdout.flush()
        if scanner_thread and not alive:
            log("Thread died! Restarting...")
            sys.stdout.flush()
            scanner_running = False
            start_scanner()

log("Starting monitor thread...")
sys.stdout.flush()
monitor = threading.Thread(target=monitor_thread, daemon=True)
monitor.start()
log(f"Monitor thread started, alive: {monitor.is_alive()}")
sys.stdout.flush()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
