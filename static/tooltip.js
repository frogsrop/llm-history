/**
 * tooltip.js — Universal `?` button component
 *
 * Usage in HTML:
 *   <span class="term">
 *     hidden state
 *     <button class="hint-btn" data-hint="Like a notebook that erases old notes when new ones are added.">?</button>
 *   </span>
 *
 * Click → expands the analogy text next to the term.
 * Second click → hides it.
 * Click outside tooltip → hides all open tooltips.
 */

(function () {
  function closeAll() {
    document.querySelectorAll(".hint-btn.open").forEach(btn => {
      btn.classList.remove("open");
      const tip = btn.parentElement.querySelector(".hint-text");
      if (tip) tip.classList.remove("visible");
    });
  }

  function initTooltips(root) {
    root = root || document;
    root.querySelectorAll(".hint-btn[data-hint]").forEach(btn => {
      // Create hint-text if not already created
      if (!btn.parentElement.querySelector(".hint-text")) {
        const tip = document.createElement("span");
        tip.className = "hint-text";
        tip.textContent = btn.dataset.hint;
        btn.parentElement.appendChild(tip);
      }

      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const isOpen = btn.classList.contains("open");

        closeAll();

        if (!isOpen) {
          btn.classList.add("open");
          const tip = btn.parentElement.querySelector(".hint-text");
          if (tip) tip.classList.add("visible");
        }
      });
    });
  }

  // Initialize on page load
  document.addEventListener("DOMContentLoaded", () => {
    initTooltips();

    // Close on outside click
    document.addEventListener("click", () => closeAll());
  });

  // Export for dynamically added elements
  window.initTooltips = initTooltips;
})();
