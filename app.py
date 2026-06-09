# app.py - Emergency fix with timeout protection
import os, sys, threading, asyncio, time, traceback
from datetime import datetime
from flask import Flask, jsonify, make_response

os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.stdout.reconfigure(line_buffering=True)

app = Flask(__name__)

scanner_running = False
scanner_thread = None
startup_logs = []
last_error = None
restart_count = 0
monitor_checks = 0
scan_count = 0
last_scan_time = None

def log(msg):
    ts = datetime.utcnow().strftime('%H:%M:%S')
    entry = f"{ts} {msg}"
    startup_logs.append(entry)
    print(f"[APP] {msg}")
    sys.stdout.flush()

def json_response(data):
    response = make_response(jsonify(data))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route("/")
def health():
    return json_response({
        "status": "alive",
        "scanner_running": scanner_running,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "restart_count": restart_count,
        "monitor_checks": monitor_checks,
        "scan_count": scan_count,
        "last_scan": str(last_scan_time) if last_scan_time else None,
        "last_error": last_error,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/status")
def scanner_status():
    return json_response({
        "scanner_running": scanner_running,
        "thread_alive": scanner_thread.is_alive() if scanner_thread else False,
        "restart_count": restart_count,
        "monitor_checks": monitor_checks,
        "scan_count": scan_count,
        "last_scan": str(last_scan_time) if last_scan_time else None,
        "last_error": last_error,
        "timestamp": datetime.utcnow().isoformat()
    })

def run_scanner():
    global scanner_running, last_error, scan_count, last_scan_time
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, base_dir)

        from scanner.v6_scanner import V6Scanner
        scanner = V6Scanner(os.path.join(base_dir, "config", "mm_config.yaml"))

        # Test Telegram
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(scanner.tg._send("🔄 Bot restarted - " + datetime.utcnow().strftime('%H:%M')))
            loop.close()
        except Exception as e:
            log(f"Telegram test failed: {e}")

        scanner_running = True
        interval = int(os.environ.get("SCAN_INTERVAL", "1800"))

        while True:
            try:
                log(f"Starting scan #{scan_count + 1}...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run scan with 5-minute timeout
                future = asyncio.ensure_future(scanner.scan(), loop=loop)
                loop.run_until_complete(asyncio.wait_for(future, timeout=300))

                scan_count += 1
                last_scan_time = datetime.utcnow()
                log(f"Scan #{scan_count} completed at {last_scan_time.strftime('%H:%M')}")
                loop.close()

                # Heartbeat every 10 scans
                if scan_count % 10 == 0:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(scanner._send_heartbeat())
                        loop.close()
                    except Exception as e:
                        log(f"Heartbeat failed: {e}")

            except asyncio.TimeoutError:
                log("Scan timed out! Restarting loop...")
                try:
                    loop.close()
                except:
                    pass
            except Exception as e:
                last_error = str(e)
                log(f"Scan error: {e}")
                try:
                    loop.close()
                except:
                    pass

            log(f"Sleeping {interval}s...")
            time.sleep(interval)

    except Exception as e:
        last_error = str(e)
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        scanner_running = False

def start_scanner():
    global scanner_thread, restart_count
    restart_count += 1
    log(f"Starting scanner (restart #{restart_count})...")
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()

start_scanner()

def monitor():
    global scanner_thread, monitor_checks
    while True:
        time.sleep(60)
        monitor_checks += 1
        alive = scanner_thread.is_alive() if scanner_thread else False

        if not alive:
            log(f"Thread dead! Restarting...")
            scanner_running = False
            start_scanner()
        elif monitor_checks % 5 == 0:
            log(f"Check #{monitor_checks}, alive={alive}, scans={scan_count}, last_scan={last_scan_time}")

threading.Thread(target=monitor, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
