import os, sys, subprocess, time, json
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# Status file for cross-process communication
STATUS_FILE = "/tmp/bot_status.json"

def read_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "running": False,
            "scan_count": 0,
            "last_scan": None,
            "error": None,
            "started": None
        }

def write_status(data):
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

@app.route("/")
def health():
    status = read_status()
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        **status,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health")
def detailed_health():
    return jsonify({"status": "running", "timestamp": datetime.utcnow().isoformat()})

# Start scanner as completely separate subprocess
def start_scanner_subprocess():
    import threading
    def run_scanner():
        while True:
            try:
                write_status({
                    "running": True,
                    "scan_count": 0,
                    "last_scan": None,
                    "error": None,
                    "started": datetime.utcnow().isoformat()
                })
                
                # Run scanner in subprocess
                result = subprocess.run(
                    [sys.executable, "-c", """
import sys, os
sys.path.insert(0, '/opt/render/project/src')
os.chdir('/opt/render/project/src')

import asyncio
import time
from scanner.v6_scanner import V6Scanner
import json
from datetime import datetime

STATUS_FILE = "/tmp/bot_status.json"

def write_status(data):
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

try:
    scanner = V6Scanner("config/mm_config.yaml")
    write_status({
        "running": True,
        "scan_count": 0,
        "last_scan": None,
        "error": None,
        "started": datetime.utcnow().isoformat()
    })
    
    # Test Telegram
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(scanner.tg._send("🔄 Bot restarted - " + datetime.utcnow().strftime('%H:%M')))
        loop.close()
    except Exception as e:
        print(f"[Scanner] Telegram test failed: {e}")
    
    # Run scans
    scan_count = 0
    interval = int(os.environ.get("SCAN_INTERVAL", "1800"))
    
    while True:
        try:
            print(f"[Scanner] Starting scan #{scan_count + 1}...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(scanner.scan())
            scan_count += 1
            last_scan = datetime.utcnow().isoformat()
            print(f"[Scanner] Scan #{scan_count} completed at {last_scan}")
            loop.close()
            
            write_status({
                "running": True,
                "scan_count": scan_count,
                "last_scan": last_scan,
                "error": None,
                "started": datetime.utcnow().isoformat()
            })
            
            # Heartbeat every 10 scans
            if scan_count % 10 == 0:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(scanner._send_heartbeat())
                    loop.close()
                except Exception as e:
                    print(f"[Scanner] Heartbeat failed: {e}")
            
        except Exception as e:
            print(f"[Scanner] Scan error: {e}")
            import traceback
            traceback.print_exc()
            write_status({
                "running": True,
                "scan_count": scan_count,
                "last_scan": datetime.utcnow().isoformat(),
                "error": str(e),
                "started": datetime.utcnow().isoformat()
            })
        
        print(f"[Scanner] Sleeping {interval}s...")
        time.sleep(interval)
        
except Exception as e:
    print(f"[Scanner] FATAL: {e}")
    import traceback
    traceback.print_exc()
    write_status({
        "running": False,
        "scan_count": 0,
        "last_scan": None,
        "error": str(e),
        "started": None
    })
"""],
                    capture_output=True,
                    text=True,
                    timeout=None
                )
                
                # If subprocess exits, log output and restart
                print(f"[Scanner] Subprocess exited with code: {result.returncode}")
                if result.stdout:
                    print(f"[Scanner] stdout: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"[Scanner] stderr: {result.stderr[-500:]}")
                    
            except Exception as e:
                print(f"[Scanner] Subprocess error: {e}")
                import traceback
                traceback.print_exc()
            
            print("[Scanner] Restarting subprocess in 30s...")
            time.sleep(30)
    
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()

# Start scanner after Flask is ready
start_scanner_subprocess()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
