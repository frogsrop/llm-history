/**
 * nav.js — Sidebar navigation, progress tracking, keyboard shortcuts
 *
 * Keyboard shortcuts:
 *   → / ←   navigate between modules
 *   Space    dispatches the custom 'animate' event — each module listens itself
 */

const MODULES = [
  { path: "/",         label: "Home",            epoch: null },
  { path: "/module/1", label: "N-gram",          epoch: "ngram" },
  { path: "/module/2", label: "RNN / LSTM",      epoch: "rnn" },
  { path: "/module/3", label: "Embeddings",      epoch: "embeddings" },
  { path: "/module/4", label: "LLM Era",         epoch: "llm" },
  { path: "/module/5", label: "Comparison",      epoch: null },
];

const STORAGE_KEY = "llm_explainer_visited";

function getVisited() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function markVisited(path) {
  const visited = getVisited();
  if (!visited.includes(path)) {
    visited.push(path);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(visited));
  }
}

function currentIndex() {
  const path = window.location.pathname;
  return MODULES.findIndex(m => m.path === path);
}

function navigate(delta) {
  const idx = currentIndex();
  const next = idx + delta;
  if (next >= 0 && next < MODULES.length) {
    window.location.href = MODULES[next].path;
  }
}

function renderSidebar() {
  const sidebar = document.getElementById("sidebar");
  if (!sidebar) return;

  const visited = getVisited();
  const curIdx = currentIndex();

  // Logo
  const logo = document.createElement("div");
  logo.className = "sidebar-logo";
  logo.textContent = "LLM Evolution";
  sidebar.appendChild(logo);

  // Nav list
  const ul = document.createElement("ul");
  ul.className = "nav-list";

  MODULES.forEach((mod, idx) => {
    const li = document.createElement("li");
    li.className = "nav-item";
    if (mod.epoch) li.dataset.epoch = mod.epoch;

    const a = document.createElement("a");
    a.className = "nav-link" + (idx === curIdx ? " active" : "");
    a.href = mod.path;

    const dot = document.createElement("span");
    dot.className = "nav-dot" +
      (idx === curIdx ? " active" : visited.includes(mod.path) ? " done" : "");

    a.appendChild(dot);
    a.appendChild(document.createTextNode(mod.label));
    li.appendChild(a);
    ul.appendChild(li);
  });

  sidebar.appendChild(ul);

  // Progress
  const donePaths = MODULES.slice(1).map(m => m.path);
  const doneCount = donePaths.filter(p => visited.includes(p)).length;
  const pct = Math.round((doneCount / (MODULES.length - 1)) * 100);

  const progressSection = document.createElement("div");
  progressSection.className = "sidebar-progress";
  progressSection.innerHTML = `
    <div class="progress-label">Studied ${doneCount} / ${MODULES.length - 1}</div>
    <div class="progress-bar-wrap">
      <div class="progress-bar-fill" style="width:${pct}%"></div>
    </div>
  `;
  sidebar.appendChild(progressSection);

  // Prev / Next buttons
  const navBtns = document.createElement("div");
  navBtns.className = "nav-buttons";

  const prevBtn = document.createElement("button");
  prevBtn.className = "btn";
  prevBtn.innerHTML = "← Back";
  prevBtn.disabled = curIdx <= 0;
  prevBtn.addEventListener("click", () => navigate(-1));

  const nextBtn = document.createElement("button");
  nextBtn.className = "btn btn-primary";
  nextBtn.innerHTML = "Next →";
  nextBtn.disabled = curIdx >= MODULES.length - 1;
  nextBtn.addEventListener("click", () => navigate(1));

  navBtns.appendChild(prevBtn);
  navBtns.appendChild(nextBtn);
  sidebar.appendChild(navBtns);

  // Seed control
  const seedSection = document.createElement("div");
  seedSection.style.cssText = "margin-top:20px; padding-top:16px; border-top:1px solid var(--border);";
  seedSection.innerHTML = `
    <div style="font-size:12px; color:var(--text-muted); margin-bottom:8px;">Seed</div>
    <div style="display:flex; align-items:center; gap:6px;">
      <input type="number" id="global-seed" value="${getSeed()}" min="-1"
             style="width:68px; background:var(--bg); border:1px solid var(--border);
                    border-radius:var(--radius); color:var(--text); padding:5px 8px;
                    font-size:13px; outline:none; text-align:center; flex-shrink:0;">
      <span style="font-size:11px; color:var(--text-muted);">-1=rand</span>
    </div>
  `;
  sidebar.appendChild(seedSection);

  sidebar.querySelector("#global-seed").addEventListener("change", (e) => {
    setSeed(parseInt(e.target.value));
  });
}

const SEED_KEY = "llm_explainer_seed";

function getSeed() {
  const v = localStorage.getItem(SEED_KEY);
  return v !== null ? parseInt(v) : 42;
}

function setSeed(val) {
  localStorage.setItem(SEED_KEY, val);
}

// Mark current page as visited on load
document.addEventListener("DOMContentLoaded", () => {
  markVisited(window.location.pathname);
  renderSidebar();
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  // Don't intercept when typing in inputs
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  if (e.key === "ArrowRight") {
    e.preventDefault();
    navigate(1);
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    navigate(-1);
  } else if (e.key === " " || e.code === "Space") {
    e.preventDefault();
    document.dispatchEvent(new CustomEvent("animate"));
  }
});
