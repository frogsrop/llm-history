"""
test_ui.py — UI tests: clicks, sliders, animations (Playwright).
Basic tests are active. Module-specific tests are uncommented as Steps 4–8 are implemented.
"""

import pytest


def test_keyboard_navigation(page, server):
    """Arrow → navigates to /module/1."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")

    page.keyboard.press("ArrowRight")
    page.wait_for_url("**/module/1")
    assert "/module/1" in page.url


def test_keyboard_back(page, server):
    """Arrow ← on /module/1 navigates back to /."""
    page.goto(server + "/module/1")
    page.wait_for_load_state("networkidle")

    page.keyboard.press("ArrowLeft")
    page.wait_for_url(server + "/")
    assert page.url.rstrip("/") == server


def test_space_dispatches_animate(page, server):
    """Space dispatches the custom 'animate' event."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")

    # Set up a listener flag
    page.evaluate("""() => {
        window._animateFired = false;
        document.addEventListener('animate', () => { window._animateFired = true; }, { once: true });
    }""")

    page.keyboard.press("Space")
    page.wait_for_timeout(200)

    fired = page.evaluate("() => window._animateFired")
    assert fired is True


def test_hint_button_toggle(page, server):
    """? button opens and closes the hint."""
    # Inject a test element into the page
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")

    # Add a test term with a button
    page.evaluate("""() => {
        const span = document.createElement('span');
        span.className = 'term';
        span.innerHTML = 'Test <button class="hint-btn" data-hint="Test hint text">?</button>';
        document.getElementById('main-content').prepend(span);
        window.initTooltips(document.getElementById('main-content'));
    }""")

    btn = page.locator(".hint-btn").first
    hint = page.locator(".hint-text").first

    assert not hint.is_visible()
    btn.click()
    assert hint.is_visible()
    btn.click()
    assert not hint.is_visible()


# === Step 3: index.html / Timeline ===

def test_timeline_events_present(page, server):
    """Timeline contains 4 events (N-gram, RNN, Word2Vec, LLM)."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    dots = page.locator(".timeline-event")
    assert dots.count() == 4, f"Expected 4 timeline events, found {dots.count()}"


def test_timeline_click_shows_popup(page, server):
    """Clicking a timeline event opens the popup with a description."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    page.locator(".timeline-event").first.click()
    popup = page.locator(".timeline-popup.visible")
    assert popup.count() >= 1, "Popup did not appear after clicking timeline"


def test_module_cards_present(page, server):
    """The home page has cards for all 5 modules."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    cards = page.locator(".module-card")
    assert cards.count() == 5, f"Expected 5 module cards, found {cards.count()}"


def test_progress_bar_present(page, server):
    """The progress bar is rendered in the main page content."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    bar = page.locator("#main-content .progress-bar-fill")
    assert bar.count() >= 1, "Progress bar not found in #main-content"


def test_module_card_navigation(page, server):
    """Clicking a module card navigates to the correct page."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    page.locator(".module-card[href='/module/1']").click()
    page.wait_for_url("**/module/1")
    assert "/module/1" in page.url


def test_reset_button_present(page, server):
    """The 'Reset' button is present on the home page."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    btn = page.locator("#reset-btn")
    assert btn.count() == 1, "Reset button not found"
    assert btn.is_visible(), "Reset button is not visible"


def test_reset_clears_progress(page, server):
    """After clicking 'Reset' the progress bar resets to 0%."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")

    # Simulate visited modules via localStorage
    page.evaluate("""() => {
        localStorage.setItem('llm_explainer_visited',
            JSON.stringify(['/', '/module/1', '/module/2', '/module/3']));
    }""")
    page.reload()
    page.wait_for_load_state("networkidle")

    # Verify progress is non-zero
    label_before = page.locator("#progress-label").inner_text()
    assert "0 из" not in label_before, "Progress should be non-zero before reset"

    # Click reset
    page.locator("#reset-btn").click()
    page.wait_for_timeout(300)

    label_after = page.locator("#progress-label").inner_text()
    assert "0 из" in label_after, f"Expected '0 из' after reset, got: '{label_after}'"


def test_timeline_popup_has_module_link(page, server):
    """Timeline popup contains a link to the module."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    page.locator(".timeline-event").first.click()
    link = page.locator(".timeline-popup.visible .timeline-popup-btn")
    assert link.count() == 1, "Module link button not found in popup"
    href = link.get_attribute("href")
    assert "/module/" in href, f"Popup link does not point to a module: {href}"


def test_timeline_popup_closes_on_outside_click(page, server):
    """Clicking outside the popup closes it."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    page.locator(".timeline-event").first.click()
    assert page.locator(".timeline-popup.visible").count() >= 1
    # Click outside
    page.locator("h1").click()
    page.wait_for_timeout(200)
    assert page.locator(".timeline-popup.visible").count() == 0, "Popup did not close"


# === Sidebar: global seed ===
def test_sidebar_seed_input_present(page, server):
    """Sidebar contains the global seed input."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    seed_input = page.locator("#global-seed")
    assert seed_input.count() == 1, "Seed input #global-seed not found in sidebar"
    assert seed_input.get_attribute("value") is not None

def test_sidebar_seed_persists_across_pages(page, server):
    """Seed is saved in localStorage and accessible on another page."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    page.locator("#global-seed").fill("99")
    page.locator("#global-seed").dispatch_event("change")
    page.wait_for_timeout(200)

    page.goto(server + "/module/1")
    page.wait_for_load_state("networkidle")
    stored = page.evaluate("() => localStorage.getItem('llm_explainer_seed')")
    assert stored == "99", f"Seed was not saved in localStorage: {stored}"

def test_sidebar_width(page, server):
    """Sidebar is 260px wide."""
    page.goto(server + "/")
    page.wait_for_load_state("networkidle")
    width = page.evaluate("() => document.getElementById('sidebar').offsetWidth")
    assert width == 260, f"Sidebar width is {width}px, expected 260px"

# --- Module-specific UI tests (uncommented as implemented) ---

# === Step 4: N-gram ===
def test_ngram_slider_changes_table(page, server):
    """Slider N=1/2/3 re-requests the table."""
    page.goto(server + "/module/1")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    slider = page.locator("#ngram-order")
    assert slider.count() == 1, "Slider #ngram-order not found"
    slider.fill("3")
    page.wait_for_timeout(800)
    table = page.locator("#table-container table")
    assert table.count() > 0, "Table did not appear after slider change"

def test_ngram_generate_button(page, server):
    """'Generate' button shows the result."""
    page.goto(server + "/module/1")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    page.locator("#generate-btn").click()
    page.wait_for_timeout(1200)
    result = page.locator("#generation-result")
    text = result.inner_text()
    assert text.strip() != "" and "Загрузка" not in text, f"Generation result is empty or stuck: '{text}'"

def test_ngram_corpus_displayed(page, server):
    """The corpus is displayed on the page."""
    page.goto(server + "/module/1")
    page.wait_for_load_state("networkidle")
    corpus_block = page.locator(".corpus-block")
    assert corpus_block.count() >= 1
    assert "кот" in corpus_block.first.inner_text()

# === Step 5: RNN ===
def test_rnn_page_loads(page, server):
    page.goto(server + "/module/2")
    page.wait_for_load_state("networkidle")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    assert errors == [], f"JS errors: {errors}"

def test_rnn_hidden_slider_present(page, server):
    page.goto(server + "/module/2")
    page.wait_for_load_state("networkidle")
    slider = page.locator("#hidden-size")
    assert slider.count() == 1, "Slider #hidden-size not found"

def test_rnn_generate_button(page, server):
    page.goto(server + "/module/2")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)
    page.locator("#generate-btn").click()
    page.wait_for_timeout(1500)
    result = page.locator("#generation-result")
    text = result.inner_text()
    assert text.strip() != "" and "Нажмите" not in text  # checks result is not the pre-generation placeholder

def test_rnn_lstm_toggle(page, server):
    page.goto(server + "/module/2")
    page.wait_for_load_state("networkidle")
    # switch to LSTM
    page.locator("#btn-lstm").click()
    page.wait_for_timeout(300)
    active = page.locator("#btn-lstm.active")
    assert active.count() == 1, "LSTM button did not become active"

def test_rnn_svg_diagram_present(page, server):
    page.goto(server + "/module/2")
    page.wait_for_load_state("networkidle")
    svg = page.locator(".diagram-svg")
    assert svg.count() >= 1, "SVG diagram not found"

# === Step 6: Embeddings ===
# def test_embeddings_canvas_not_empty(page, server):
#     page.goto(server + "/module/3")
#     page.wait_for_timeout(1500)
#     not_empty = page.evaluate("""() => {
#         const canvas = document.querySelector('canvas');
#         if (!canvas) return false;
#         const ctx = canvas.getContext('2d');
#         const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
#         return data.some(v => v !== 0);
#     }""")
#     assert not_empty

# === Step 7: LLM ===
# def test_llm_toggle_attention(page, server):
#     page.goto(server + "/module/4")
#     toggle = page.locator("input[type='checkbox'][name='attention']")
#     toggle.check()
#     page.wait_for_timeout(2000)
#     heatmap = page.locator("canvas")
#     assert heatmap.count() > 0
