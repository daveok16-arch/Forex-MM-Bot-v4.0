#!/usr/bin/env python3
import os, sys, asyncio, time, json, traceback
from datetime import datetime

sys.path.insert(0, '/opt/render/project/src')
os.chdir('/opt/render/project/src')

STATUS_FILE = "/tmp/bot_status.json"

def write_status(data):
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

def main():
    try:
        from scanner.v6_scanner import V6Scanner
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
                
                if scan_count % 10 == 0:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.close()
                    except Exception as e:
                        print(f"[Scanner] Heartbeat failed: {e}")
                
            except Exception as e:
                print(f"[Scanner] Scan error: {e}")
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
        traceback.print_exc()
        write_status({
            "running": False,
            "scan_count": 0,
            "last_scan": None,
            "error": str(e),
            "started": None
        })

if __name__ == "__main__":
    main()
