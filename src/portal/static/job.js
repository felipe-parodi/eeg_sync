// Job page: poll /jobs/<id>/status.json while the job runs and update the
// stage list and message in place. When the job stops running (waiting for
// labels, done, or failed) the page reloads once to show the next action.
(function () {
  const root = document.getElementById("job");
  if (!root) return;
  const url = "/jobs/" + encodeURIComponent(root.dataset.jobId) + "/status.json";
  const STATE_TEXT = { done: " (done)", current: " (now)", failed: " (stopped here)", todo: "" };

  function render(job) {
    document.getElementById("job-message").textContent = job.message;
    job.stages.forEach((stage) => {
      const li = root.querySelector('.stage[data-stage="' + stage.key + '"]');
      if (!li) return;
      li.className = "stage stage--" + stage.status;
      li.querySelector(".stage__state").textContent = STATE_TEXT[stage.status] || "";
    });
    const cams = document.getElementById("job-cameras");
    cams.replaceChildren(
      ...job.cameras.map((cam) => {
        const li = document.createElement("li");
        li.textContent = cam.message;
        return li;
      })
    );
    cams.hidden = job.cameras.length === 0;
  }

  async function poll() {
    try {
      const res = await fetch(url, { credentials: "same-origin" });
      if (res.ok) {
        const job = await res.json();
        if (!job.polling || job.state !== root.dataset.state) {
          window.location.reload();
          return;
        }
        render(job);
      }
    } catch (err) {
      // Network blip: try again on the next tick.
    }
    setTimeout(poll, 4000);
  }
  setTimeout(poll, 4000);
})();
