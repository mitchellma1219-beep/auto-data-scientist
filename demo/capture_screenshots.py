#!/usr/bin/env python3
"""Capture screenshots of the dashboard with a headless browser, and check it stays local.

    pip install -e ".[ui,demo]"            # adds Playwright; Chrome or Chromium must exist
    python demo/capture_screenshots.py     # writes demo/screenshots/*.png

The script starts the dashboard on a free localhost port, walks through the sections with
the built-in synthetic samples, saves one PNG per step and stops the server. It records
every request the page makes and exits with an error if one goes to another host: the
dashboard is meant to work without a network connection.

This is optional tooling for the README, not part of the test suite: it needs a browser,
and it depends on Streamlit's page structure, which can change between versions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Page, sync_playwright

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app" / "streamlit_app.py"
WIDTH = 1440


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_dashboard(port: int) -> subprocess.Popen[bytes]:
    server = subprocess.Popen(  # noqa: S603 - fixed arguments, no shell
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port),
         "--server.headless", "true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        env={**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"})
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("the dashboard exited while starting")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                if response.read() == b"ok":
                    return server
        except OSError:
            time.sleep(0.5)
    server.terminate()
    raise RuntimeError("the dashboard did not become healthy within 120 s")


def settle(page: Page, pause_ms: int = 1500) -> None:
    """Wait until Streamlit has finished a rerun and the charts have drawn."""
    page.wait_for_timeout(300)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180_000)
    page.wait_for_timeout(pause_ms)


def capture(base: str, out: Path, browser_path: str | None) -> tuple[list[Path], list[str]]:
    """Walk through the dashboard. Returns the files written and every host contacted."""
    shots: list[Path] = []
    hosts: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=browser_path, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": WIDTH, "height": 900})
        page.on("request", lambda request: hosts.append(urlsplit(request.url).netloc))

        def shot(name: str, height: int = 900) -> None:
            page.set_viewport_size({"width": WIDTH, "height": height})
            page.wait_for_timeout(700)
            path = out / f"{name}.png"
            page.screenshot(path=str(path))
            shots.append(path)

        def section(name: str) -> None:
            page.get_by_text(name, exact=True).first.click()
            settle(page)

        def sample(title: str) -> None:
            page.locator("[data-testid=stSidebar] [data-testid=stSelectbox]").first.click()
            page.get_by_role("option", name=title).click()
            settle(page, 2000)

        page.goto(base)
        page.wait_for_selector("text=Quality score components", timeout=120_000)
        settle(page, 2500)
        shot("data", 1000)

        section("Explore")
        page.get_by_role("tab", name="Cohort retention").click()
        settle(page)
        shot("explore", 1000)

        section("Ask")
        page.get_by_label("Question about one table").fill("top 2 regions by total revenue")
        page.get_by_role("button", name="Translate and run").click()
        settle(page)
        shot("ask", 1000)
        page.get_by_text("SQL", exact=True).first.click()
        settle(page)
        page.get_by_label("SQL (one SELECT statement, DuckDB dialect)").fill(
            "SELECT * FROM read_csv('/etc/passwd')")
        page.get_by_role("button", name="Run query").click()
        settle(page)
        shot("ask_refused", 760)

        section("Forecast")
        page.get_by_role("button", name="Forecast", exact=True).click()
        settle(page, 3000)
        shot("forecast", 1000)

        section("Report")
        page.get_by_role("button", name="Build report").click()
        settle(page, 3000)
        shot("report", 1100)

        sample("Subscription churn")
        section("Predict")
        page.get_by_role("button", name="Train and evaluate").click()
        settle(page, 3000)
        shot("predict", 1250)
        browser.close()
    return shots, hosts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "screenshots")
    parser.add_argument("--browser", default=shutil.which("google-chrome")
                        or shutil.which("chromium") or shutil.which("chromium-browser"),
                        help="path to Chrome or Chromium (default: the one on PATH, else the "
                             "browser Playwright installed)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = start_dashboard(port)
    try:
        shots, hosts = capture(f"http://localhost:{port}", args.out, args.browser)
    finally:
        server.terminate()
        server.wait(timeout=30)
    for path in shots:
        print(f"wrote {path}")
    elsewhere = sorted({h for h in hosts if h not in (f"localhost:{port}", f"127.0.0.1:{port}")})
    print(f"{len(hosts)} requests, all to localhost:{port}" if not elsewhere
          else f"{len(hosts)} requests, including other hosts: {elsewhere}")
    return 1 if elsewhere else 0


if __name__ == "__main__":
    raise SystemExit(main())
