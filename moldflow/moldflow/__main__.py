"""Entry point: `moldflow` or `python -m moldflow` starts the local server and opens the browser."""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moldflow", description="Injection molding quick-simulation tool")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    ap.add_argument("--reload", action="store_true", help="dev: auto-reload on code changes")
    args = ap.parse_args(argv)

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"moldflow running at {url}  (Ctrl+C to stop)")
    uvicorn.run("moldflow.server:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
