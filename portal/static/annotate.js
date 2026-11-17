// Parent/child anchor labeling page.
// The CSRF token and job id come from data attributes on #annotate-form.
(function () {
  const form = document.getElementById("annotate-form");
  const CSRF_TOKEN = form.dataset.csrfToken;
  const JOB_ID = form.dataset.jobId;
  const state = {};

  function setBtnState(card, choice) {
    card.querySelectorAll("button").forEach((b) => b.classList.remove("chosen"));
    if (choice) choice.classList.add("chosen");
    card.querySelector(".anchor-state").textContent = choice
      ? choice.classList.contains("skip-btn")
        ? "skipped"
        : "parent=" + choice.dataset.parent
      : "unlabeled";
  }

  document.querySelectorAll(".anchor-card").forEach((card) => {
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
    const payload = {};
    Object.keys(state).forEach((cam) => {
      payload[cam] = Object.values(state[cam]);
    });
    const res = await fetch(ev.target.action, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-csrf-token": CSRF_TOKEN },
      body: JSON.stringify({ annotations: payload }),
    });
    if (res.ok) {
      window.location = "/jobs/" + JOB_ID;
    } else {
      const t = await res.text();
      alert("Submit failed: " + t);
    }
  });
})();
