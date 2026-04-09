"""
conftest.py — test fixtures.

Starts FastAPI via uvicorn subprocess on port 8000,
provides an httpx client and a Playwright browser.

Requirements:
  uv sync
  playwright install chromium
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

BASE_DIR = Path(__file__).parent.parent
SERVER_URL = "http://localhost:8000"
STARTUP_TIMEOUT = 15  # seconds (uvicorn starts fast; RNN trains in background)


@pytest.fixture(scope="session")
def server():
    """Starts uvicorn, waits for readiness, stops it after tests."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until the server is up
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.3)
    else:
        proc.terminate()
        raise RuntimeError("Server did not start within {} seconds".format(STARTUP_TIMEOUT))

    # Wait for RNN models to be ready (background training ~3 min)
    RNN_TIMEOUT = 360
    rnn_deadline = time.time() + RNN_TIMEOUT
    print(f"\n[conftest] Waiting for RNN models (up to {RNN_TIMEOUT}s)...", flush=True)
    rnn_ready = False
    while time.time() < rnn_deadline:
        try:
            r = httpx.get(f"{SERVER_URL}/api/rnn/status", timeout=2)
            if r.status_code == 200 and r.json().get("ready"):
                rnn_ready = True
                break
        except Exception as e:
            pass
        time.sleep(3)

    if rnn_ready:
        print(f"[conftest] RNN models ready after {int(time.time() - (rnn_deadline - RNN_TIMEOUT))}s", flush=True)
    else:
        print(f"[conftest] WARNING: RNN models NOT ready after {RNN_TIMEOUT}s — API tests will fail", flush=True)

    # Wait for embeddings model to be ready (fastText ~7GB, can take a while)
    EMB_TIMEOUT = 600
    emb_deadline = time.time() + EMB_TIMEOUT
    print(f"\n[conftest] Waiting for embeddings model (up to {EMB_TIMEOUT}s)...", flush=True)
    emb_ready = False
    while time.time() < emb_deadline:
        try:
            r = httpx.get(f"{SERVER_URL}/api/embeddings/status", timeout=2)
            if r.status_code == 200 and r.json().get("ready"):
                emb_ready = True
                break
        except Exception:
            pass
        time.sleep(3)

    if emb_ready:
        print(f"[conftest] Embeddings model ready after {int(time.time() - (emb_deadline - EMB_TIMEOUT))}s", flush=True)
    else:
        print(f"[conftest] WARNING: Embeddings model NOT ready after {EMB_TIMEOUT}s — embeddings tests will fail", flush=True)

    # Wait for LLM-era model to be ready (Qwen2.5-3B, can take a while on first load)
    LLM_TIMEOUT = 600
    llm_deadline = time.time() + LLM_TIMEOUT
    print(f"\n[conftest] Waiting for LLM-era model (up to {LLM_TIMEOUT}s)...", flush=True)
    llm_ready = False
    while time.time() < llm_deadline:
        try:
            r = httpx.get(f"{SERVER_URL}/api/llm-era/status", timeout=2)
            data = r.json()
            if r.status_code == 200 and data.get("ready") and data.get("seq2seq_ready"):
                llm_ready = True
                break
        except Exception:
            pass
        time.sleep(3)

    if llm_ready:
        print(f"[conftest] LLM-era model ready after {int(time.time() - (llm_deadline - LLM_TIMEOUT))}s", flush=True)
    else:
        print(f"[conftest] WARNING: LLM-era model NOT ready after {LLM_TIMEOUT}s — LLM tests will fail", flush=True)

    yield SERVER_URL

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def client(server):
    """httpx client with server base URL."""
    with httpx.Client(base_url=server, timeout=60) as c:
        yield c


@pytest.fixture(scope="session")
def browser_context(server):
    """Playwright browser + context. Requires: playwright install chromium."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not installed")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        yield context
        context.close()
        browser.close()


@pytest.fixture
def page(browser_context):
    """New page for each test."""
    p = browser_context.new_page()
    yield p
    p.close()
