/**
 * utils.js — Shared JS utilities: animateFlow(), fetch wrappers
 */

/**
 * animateFlow(elements, options)
 * Sequentially highlights DOM elements with a delay.
 *
 * @param {Element[]} elements  - list of elements to animate
 * @param {object}    options
 *   @param {number}  options.delay   - delay between elements (ms), default 400
 *   @param {string}  options.cls     - CSS class, default 'token-highlight'
 *   @param {boolean} options.keep    - keep class on the final element, default true
 *   @param {Function} options.onStep - callback(element, index) on each step
 *   @param {Function} options.onDone - callback on completion
 */
function animateFlow(elements, options = {}) {
  const {
    delay = (window.animDelay ?? 400),
    cls = "token-highlight",
    keep = true,
    onStep = null,
    onDone = null,
  } = options;

  let i = 0;

  function step() {
    if (i > 0) {
      const prev = elements[i - 1];
      if (!keep || i < elements.length) {
        prev.classList.remove(cls);
      }
    }

    if (i >= elements.length) {
      if (onDone) onDone();
      return;
    }

    const el = elements[i];
    el.classList.add(cls);
    if (onStep) onStep(el, i);
    i++;
    setTimeout(step, delay);
  }

  step();
}

/**
 * apiFetch(path, options)
 * Wrapper over fetch for API requests.
 * Automatically parses JSON, throws on non-2xx responses.
 *
 * @param {string} path     - path relative to origin, e.g. '/api/ngram/table?n=2'
 * @param {object} options  - standard fetch options
 * @returns {Promise<any>}
 */
async function apiFetch(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      msg = data.detail || JSON.stringify(data);
    } catch {}
    throw new Error(msg);
  }

  return resp.json();
}

/**
 * apiPost(path, body)
 * Shorthand for a POST request with a JSON body.
 */
async function apiPost(path, body) {
  return apiFetch(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * showError(container, message)
 * Displays an error message in a result-box.
 */
function showError(container, message) {
  if (!container) return;
  container.innerHTML = `<span style="color:var(--epoch-llm)">Error: ${escapeHtml(message)}</span>`;
}

/**
 * showLoading(container)
 * Shows a loading indicator.
 */
function showLoading(container) {
  if (!container) return;
  container.innerHTML = `<span style="color:var(--text-muted)">Loading…</span>`;
}

/**
 * escapeHtml(str)
 * Escapes HTML special characters.
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * debounce(fn, ms)
 * Delays invocation of fn until ms milliseconds after the last call.
 */
function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
