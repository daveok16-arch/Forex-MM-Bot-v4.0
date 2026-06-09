import os, json
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
