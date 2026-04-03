"""
Local trader dashboard — http://127.0.0.1:5000

  python -m src.broker_dashboard.app
  python -m src.main --broker-dashboard
  python -m src.main --dashboard   (also starts trading + opens browser)
"""
from __future__ import annotations

import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, Response, render_template

from src.broker_dashboard.service import build_full_state, build_legacy_dashboard_data_payload

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=_TEMPLATE_DIR)


@app.route("/")
def index():
    return render_template("trader.html")


@app.route("/api/summary")
def api_summary():
    s = build_full_state()
    return Response(json.dumps(s, default=str), mimetype="application/json")


@app.route("/api/dashboard-data")
def api_dashboard_data():
    """Legacy alias: same port as Trade Desk; fills shape expected by old dashboard UIs."""
    payload = build_legacy_dashboard_data_payload()
    return Response(json.dumps(payload, default=str), mimetype="application/json")


def run_app(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    print("  Trade Desk: http://127.0.0.1:5000")
    run_app()
