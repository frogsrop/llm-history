"""
test_smoke.py — basic smoke tests: server starts, pages open correctly.
"""

import pytest


PAGES = [
    ("/", 200),
    ("/module/1", 200),
    ("/module/2", 200),
    ("/module/3", 200),
    ("/module/4", 200),
    ("/module/5", 200),
    ("/health", 200),
]


@pytest.mark.parametrize("path,expected_status", PAGES)
def test_pages_return_200(client, path, expected_status):
    """All main pages return the expected HTTP status."""
    r = client.get(path)
    assert r.status_code == expected_status, (
        f"GET {path} → {r.status_code}, expected {expected_status}"
    )


def test_index_contains_module_links(client):
    """The home page contains links to all 5 modules."""
    r = client.get("/")
    html = r.text
    for i in range(1, 6):
        assert f"/module/{i}" in html, f"Link to /module/{i} not found"


def test_static_css_served(client):
    """CSS file is served correctly."""
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers.get("content-type", "")


def test_static_js_served(client):
    """JS files are served correctly."""
    for js in ["nav.js", "utils.js", "tooltip.js"]:
        r = client.get(f"/static/{js}")
        assert r.status_code == 200, f"{js} is not being served"


def test_page_opens_in_browser(page, server):
    """Playwright: home page opens without JS errors."""
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(server + "/")
    page.wait_for_load_state("networkidle")

    assert page.title() != "", "Page title is empty"
    assert errors == [], f"JS errors on page: {errors}"


def test_sidebar_rendered(page, server):
    """Playwright: sidebar is rendered with the expected links."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")

    sidebar = page.locator("#sidebar")
    assert sidebar.count() == 1, "Sidebar not found"

    # Verify module links are present in the sidebar
    for i in range(1, 6):
        link = page.locator(f"#sidebar a[href='/module/{i}']")
        assert link.count() >= 1, f"Link /module/{i} not found in sidebar"
