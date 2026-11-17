// Parent/child anchor labeling page.
// The CSRF token and job id come from data attributes on #annotate-form.
(function () {
  const form = document.getElementById("annotate-form");
  const CSRF_TOKEN = form.dataset.csrfToken;
  const JOB_ID = form.dataset.jobId;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const cards = Array.from(document.querySelectorAll(".anchor-card"));
  const progress = document.getElementById("annotate-progress");
  const state = {};

  function updateProgress() {
    let done = 0;
    let skipped = 0;
    Object.values(state).forEach((cam) =>
      Object.values(cam).forEach((entry) => {
        done += 1;
        if (entry.skipped) skipped += 1;
      })
    );
    progress.textContent =
      done + " of " + cards.length + " labeled" + (skipped ? " (" + skipped + " skipped)" : "");
  }

  // Outline the chosen parent's box on the frame (the colored boxes are
  // already drawn into the image). Coordinates are image pixels.
  function markParentBox(card, btn) {
    const svg = card.querySelector(".anchor-frame svg");
    const img = card.querySelector(".anchor-frame img");
    if (!svg || !img) return;
    svg.replaceChildren();
    if (!btn || !btn.dataset.bbox || !img.naturalWidth) return;
    const [x1, y1, x2, y2] = btn.dataset.bbox.split(",").map(Number);
    svg.setAttribute("viewBox", "0 0 " + img.naturalWidth + " " + img.naturalHeight);
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("class", "box-parent");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", x2 - x1);
    rect.setAttribute("height", y2 - y1);
    const size = Math.round(img.naturalHeight / 18);
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("class", "box-parent-label");
    label.setAttribute("x", x1 + 6);
    label.setAttribute("y", y1 + size + 4);
    label.setAttribute("font-size", size);
    label.textContent = "PARENT";
    svg.append(rect, label);
  }

  function setBtnState(card, choice) {
    card.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("chosen", b === choice);
      b.setAttribute("aria-pressed", b === choice ? "true" : "false");
    });
    const skipped = choice.classList.contains("skip-btn");
    card.classList.toggle("is-labeled", !skipped);
    card.classList.toggle("is-skipped", skipped);
    card.querySelector(".anchor-state").textContent = skipped
      ? "Skipped"
      : "Parent: track " + choice.dataset.parent;
    markParentBox(card, skipped ? null : choice);
    updateProgress();
  }

  cards.forEach((card) => {
    card.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    card.querySelectorAll(".track-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cam = btn.dataset.camera;
        const idx = btn.dataset.anchor;
        const parent = parseInt(btn.dataset.parent, 10);
        const others = Array.from(card.querySelectorAll(".track-btn"))
          .map((b) => parseInt(b.dataset.parent, 10))
          .filter((t) => t !== parent);
        state[cam] = state[cam] || {};
        state[cam][idx] = {
          anchor_idx: parseInt(idx, 10),
          parent_track_id: parent,
          child_track_id: others[0],
        };
        setBtnState(card, btn);
      });
    });
    const skip = card.querySelector(".skip-btn");
    if (skip) {
      skip.addEventListener("click", () => {
        const cam = skip.dataset.camera;
        const idx = skip.dataset.anchor;
        state[cam] = state[cam] || {};
        state[cam][idx] = { anchor_idx: parseInt(idx, 10), skipped: true };
        setBtnState(card, skip);
      });
    }
  });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const submit = form.querySelector("button[type=submit]");
    const payload = {};
    Object.keys(state).forEach((cam) => {
      payload[cam] = Object.values(state[cam]);
    });
    submit.disabled = true;
    const res = await fetch(ev.target.action, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-csrf-token": CSRF_TOKEN },
      body: JSON.stringify({ annotations: payload }),
    });
    if (res.ok) {
      window.location = "/jobs/" + JOB_ID;
      return;
    }
    let detail = await res.text();
    try {
      detail = JSON.parse(detail).detail || detail;
    } catch (err) {
      // Not JSON: show the text as is.
    }
    alert(detail);
    submit.disabled = false;
  });
})();
