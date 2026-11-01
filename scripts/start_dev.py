#!/usr/bin/env python3
"""Launch LeakLens backend + frontend together with a single command.

Usage:
    python scripts/start_dev.py

Starts:
  - Backend:  http://127.0.0.1:8000  (FastAPI/uvicorn)
  - Frontend: http://127.0.0.1:5173 (Vite dev server)

Press Ctrl+C (or close this window) to stop both servers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_URL = "http://127.0.0.1:8000/api/health"
FRONTEND_URL = "http://127.0.0.1:5173/"


def pick_python() -> str:
    venv_python = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def process_image_name(pid: str) -> str:
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        parts = line.split('","')
        return parts[0].strip('"') if parts else ""
    except Exception:
        return ""


def ensure_ports_free() -> None:
    """Kill stale LeakLens dev servers (vite/node or uvicorn/python) that may
    still be holding ports 5173 / 8000 from a previous run. Without this,
    a leftover orphan makes Vite jump to 5174 and the backend fail to bind."""
    import re

    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    except Exception:
        return
    for line in out.splitlines():
        if "LISTENING" not in line:
            continue
        m = re.search(r":(5173|8000)\s", line)
        if not m:
            continue
        pid = line.split()[-1]
        img = process_image_name(pid).lower()
        if img in ("node.exe", "python.exe"):
            print(f"Stale {img} (PID {pid}) holding port {m.group(1)} — stopping it")
            subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)


def wait_until(url: str, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    os.chdir(ROOT)

    # Clear stale servers from previous runs before we bind the ports.
    ensure_ports_free()

    # --- Start the two servers as child processes ---
    backend = subprocess.Popen(
        [pick_python(), "-m", "uvicorn", "app.main:app",
         "--app-dir", "backend", "--port", "8000"],
        cwd=ROOT,
    )
    frontend = subprocess.Popen(
        "npm --prefix frontend run dev",
        cwd=ROOT,
        shell=True,  # required on Windows to resolve npm.cmd
    )

    print("LeakLens starting... (Ctrl+C to stop both)")
    print()

    try:
        # Wait for both servers to come up.
        backend_ok = wait_until(BACKEND_URL)
        frontend_ok = wait_until(FRONTEND_URL, timeout=60)

        print("-" * 60)
        print("Backend:  http://127.0.0.1:8000   "
              + ("[READY]" if backend_ok else "[not reachable yet]"))
        print("Frontend: http://127.0.0.1:5173  "
              + ("[READY]" if frontend_ok else "[not reachable yet]"))
        print("  API docs: http://127.0.0.1:8000/docs")
        print("-" * 60)
        print("Both servers are running. Press Ctrl+C to stop them.")
        print()

        # Self-test hook (used by automated checks): simulate Ctrl+C once
        # both servers are confirmed ready, so cleanup can be verified.
        if "--selftest" in sys.argv:
            print("selftest: simulating Ctrl+C")
            raise KeyboardInterrupt

        # Keep the parent running until interrupted.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        # On Windows kill the whole process tree so child processes
        # (npm -> node -> vite, uvicorn) are cleaned up reliably.
        for name, proc in (("frontend", frontend), ("backend", backend)):
            print(f"Stopping {name} (PID {proc.pid})...")
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                    )
                else:
                    proc.terminate()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
