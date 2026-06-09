# app.py - Minimal health endpoint + separate scanner process
import os, sys, threading, asyncio, time, traceback
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# Global state (simple, never blocks)
bot_status = {
    "running": False,
    "last_scan": None,
    "scan_count": 0,
    "error": None,
    "started": datetime.utcnow().isoformat()
}

@app.route("/")
def health():
    return jsonify({
        "status": "alive",
        "bot": "forex-signal-bot-v4",
        "running": bot_status["running"],
        "last_scan": bot_status["last_scan"],
        "scan_count": bot_status["scan_count"],
        "error": bot_status["error"],
        "started": bot_status["started"],
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health")
def detailed_health():
    return jsonify({"status": "running", "timestamp": datetime.utcnow().isoformat()})

def run_scanner():
    """Run scanner in completely separate thread"""
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)
    
    try:
        from scanner.v6_scanner import V6Scanner
        scanner = V6Scanner(os.path.join(base_dir, "config", "mm_config.yaml"))
        
        # Test Telegram
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(scanner.tg._send("🔄 Bot restarted - " + datetime.utcnow().strftime('%H:%M')))
            loop.close()
        except Exception as e:
            print(f"[Scanner] Telegram test failed: {e}")
        
        bot_status["running"] = True
        interval = int(os.environ.get("SCAN_INTERVAL", "1800"))
        
        while True:
            try:
                print(f"[Scanner] Starting scan #{bot_status['scan_count'] + 1}...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run scan with 5-minute timeout
                future = asyncio.ensure_future(scanner.scan(), loop=loop)
                loop.run_until_complete(asyncio.wait_for(future, timeout=300))
                
                bot_status["scan_count"] += 1
                bot_status["last_scan"] = datetime.utcnow().isoformat()
                print(f"[Scanner] Scan #{bot_status['scan_count']} completed")
                loop.close()
                
                # Heartbeat every 10 scans
                if bot_status["scan_count"] % 10 == 0:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(scanner._send_heartbeat())
                        loop.close()
                    except Exception as e:
                        print(f"[Scanner] Heartbeat failed: {e}")
                
            except asyncio.TimeoutError:
                print("[Scanner] Scan timed out!")
                try:
                    loop.close()
                except:
n                    pass
            except Exception as e:
                bot_status["error"] = str(e)
                print(f"[Scanner] Error: {e}")
                traceback.print_exc()
                try:
                    loop.close()
                except:
                    pass
            
            print(f"[Scanner] Sleeping {interval}s...")
            time.sleep(interval)
            
    except Exception as e:
        bot_status["error"] = str(e)
        print(f"[Scanner] FATAL: {e}")
        traceback.print_exc()
        bot_status["running"] = False

# Start scanner in background thread
scanner_thread = threading.Thread(target=run_scanner, daemon=True)
scanner_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
