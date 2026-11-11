#!/usr/bin/env python3
"""Runner script for the Tagging & Notice Visualizer Flask App.

Usage:
    python tagging/flaskapp/run.py
    python tagging/flaskapp/run.py --data-dir "C:/path/to/another_datastore" --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tagging.flaskapp.app import create_app  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run Notice & Tagging Visualizer Flask Application."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default=os.environ.get("DATA_STORE_DIR", "data_store"),
        help="Path to the root data directory (default: 'data_store' or DATA_STORE_DIR env var).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port to run the Flask development server on (default: 5000).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Run in Flask debug mode (default: True).",
    )

    args = parser.parse_args()

    data_dir_resolved = Path(args.data_dir).resolve()
    print("=" * 65)
    print("  Notice & Tagging Visualizer Flask App")
    print("=" * 65)
    print(f"  Data Store Directory : {data_dir_resolved}")
    print(f"  Server URL           : http://{args.host}:{args.port}")
    print("=" * 65)

    app = create_app(data_store_dir=str(data_dir_resolved))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
