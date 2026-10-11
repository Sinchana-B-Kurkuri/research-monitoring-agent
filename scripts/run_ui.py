"""
run_ui.py
~~~~~~~~~
CLI command to run the Phase 6 FastAPI web application server.

Usage:
  python scripts/run_ui.py --port 8000
"""

import argparse
import os
import sys
import uvicorn

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Research Monitor UI server (Phase 6).")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable live reload for development")
    args = parser.parse_args()

    print(f"Starting Research Monitor web server at http://{args.host}:{args.port}")
    print("Authenticate by visiting:")
    print(f"  http://{args.host}:{args.port}/review?token=<your_access_token>")
    print(f"  http://{args.host}:{args.port}/digest?token=<your_access_token>")
    print(f"  http://{args.host}:{args.port}/audit?token=<your_access_token>")

    uvicorn.run("research_monitor.ui.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
