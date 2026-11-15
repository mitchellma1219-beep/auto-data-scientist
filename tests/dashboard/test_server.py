"""The dashboard as a real process, and static checks on what the page is allowed to do."""

import ast
import os
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

import pytest

if os.environ.get("ADS_REQUIRE_UI") != "1":      # CI sets it: there a missing extra must fail
    pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app" / "streamlit_app.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def outside_address() -> str | None:
    """This machine's address on its default route, if it has one besides loopback."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))            # TEST-NET-1: nothing is sent for UDP
            address = str(s.getsockname()[0])
    except OSError:
        return None
    return None if address.startswith("127.") else address


def fetch(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


def test_dashboard_starts_headless_answers_health_checks_and_stops(tmp_path):
    port = free_port()
    log = tmp_path / "server.log"
    env = {**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    with log.open("w") as out:
        # Started from another directory on purpose: the settings next to the script,
        # including the localhost binding, must apply wherever it is launched from.
        server = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port),
             "--server.headless", "true"],
            cwd=tmp_path, stdout=out, stderr=subprocess.STDOUT, env=env)
    try:
        health = f"http://127.0.0.1:{port}/_stcore/health"
        deadline, body = time.monotonic() + 120, b""
        while time.monotonic() < deadline:
            assert server.poll() is None, f"the server exited early:\n{log.read_text()}"
            try:
                status, body = fetch(health)
                if status == 200 and body == b"ok":
                    break
            except OSError:
                pass
            time.sleep(0.5)
        assert body == b"ok", f"no healthy answer within 120 s:\n{log.read_text()}"
        status, page = fetch(f"http://127.0.0.1:{port}/")
        assert status == 200 and b'<div id="root">' in page

        outside = outside_address()
        if outside is not None:                      # bound to localhost: closed elsewhere
            with pytest.raises(OSError), socket.create_connection((outside, port), timeout=3):
                pass
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)
    assert server.returncode is not None
    text = log.read_text()
    assert "Traceback" not in text and f"localhost:{port}" in text


def test_settings_next_to_the_script_keep_it_local_and_quiet():
    settings = tomllib.loads((APP.parent / ".streamlit" / "config.toml").read_text())
    assert settings["server"]["address"] == "localhost"
    assert settings["browser"]["gatherUsageStats"] is False
    assert settings["runner"]["magicEnabled"] is False
    assert settings["client"]["showErrorDetails"] == "type"
    assert {"light", "dark"} <= set(settings["theme"])
    favicon = (APP.parent / "favicon.svg").read_text()
    assert favicon.startswith("<svg") and "http://www.w3.org/2000/svg" in favicon
    assert "href" not in favicon and "<script" not in favicon


def calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            yield (func.id if isinstance(func, ast.Name) else None,
                   func.attr if isinstance(func, ast.Attribute) else None, node)


def test_nothing_executes_generated_code_or_renders_raw_html():
    sources = [APP, *sorted((ROOT / "src").rglob("*.py"))]
    assert len(sources) > 15
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"pickle", "marshal", "shelve", "subprocess", "ctypes", "code",
                               "runpy"}, path
        for name, attr, node in calls(tree):
            assert name not in {"eval", "exec", "compile", "__import__"}, (path, node.lineno)
            assert attr not in {"system", "popen", "spawn", "read_pickle", "to_pickle",
                                "import_module"}, (path, node.lineno)
            for keyword in node.keywords:
                assert keyword.arg not in {"unsafe_allow_html", "unsafe_allow_javascript"}, (
                    path, node.lineno)
    app = APP.read_text(encoding="utf-8")
    assert ":material/" in app                       # icons in the page come from the bundled font
    assert 'page_icon=":material' not in app          # the page icon would be fetched from a CDN
    assert "http://" not in app and "https://" not in app
