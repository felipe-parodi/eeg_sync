// Gaze labeling page. Reads its data from the #gaze-data JSON block.
(function () {
  const D = JSON.parse(document.getElementById("gaze-data").textContent);
  const COLORS = { parent: "#1f8a4c", child: "#d9730d" };
  const NAMES = { parent: "Parent", child: "Child" };
  const CAMERAS = { camera_a: "Video A", camera_b: "Video B" };
  const state = {}; // "cam|frame|person" -> {target, point}
  for (const [cam, labels] of Object.entries(D.existing)) {
    for (const l of labels) {
      state[cam + "|" + l.frame_idx + "|" + l.person] = { target: l.target, point: l.point };
    }
  }
  let armed = null;
  const cards = document.getElementById("cards");

  function progress() {
    const total = Object.values(D.items).reduce((n, a) => n + a.length * 2, 0);
    const done = Object.values(state).filter((v) => v.target).length;
    document.getElementById("progress").textContent = done + " of " + total + " answers";
  }

  // Tint a card once both people have an answer (presentation only).
  function refreshCard(card, cam, frame) {
    const answered = ["parent", "child"].filter((p) => {
      const s = state[cam + "|" + frame + "|" + p];
      return s && s.target;
    }).length;
    card.classList.toggle("is-labeled", answered === 2);
    card.classList.toggle("is-partial", answered === 1);
  }

  function clock(t) {
    return Math.floor(t / 60) + ":" + String(Math.round(t % 60)).padStart(2, "0");
  }

  for (const [cam, items] of Object.entries(D.items)) {
    const h = document.createElement("h3");
    h.textContent = CAMERAS[cam] || cam;
    cards.appendChild(h);
    items.forEach((it, i) => {
      const card = document.createElement("div");
      card.className = "gcard";
      const cap = document.createElement("div");
      cap.className = "gcard__caption";
      cap.textContent =
        (CAMERAS[cam] || cam) + ", frame " + (i + 1) + " of " + items.length + " (" + it.block + ", " + clock(it.timestamp_s) + ")";
      const canvas = document.createElement("canvas");
      const img = new Image();
      img.src = D.frames_url + cam + "/" + it.image_name;
      function draw() {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const c = canvas.getContext("2d");
        c.drawImage(img, 0, 0);
        for (const p of ["parent", "child"]) {
          const b = it.boxes[p];
          if (!b) continue;
          c.strokeStyle = COLORS[p];
          c.lineWidth = 3;
          c.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
          const s = state[cam + "|" + it.frame_idx + "|" + p];
          if (s && s.point) {
            const x = s.point[0] * canvas.width;
            const y = s.point[1] * canvas.height;
            c.fillStyle = COLORS[p];
            c.beginPath();
            c.arc(x, y, 7, 0, 2 * Math.PI);
            c.fill();
            c.strokeStyle = "#fff";
            c.lineWidth = 2;
            c.stroke();
          }
        }
      }
      img.onload = draw;
      canvas.addEventListener("click", (ev) => {
        if (!armed || armed.card !== card) return;
        const r = canvas.getBoundingClientRect();
        const key = cam + "|" + it.frame_idx + "|" + armed.person;
        state[key] = state[key] || {};
        state[key].point = [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
        armed.button.classList.remove("armed");
        armed = null;
        draw();
      });
      card.append(cap, canvas);
      for (const p of ["parent", "child"]) {
        const row = document.createElement("div");
        row.className = "grow";
        const lab = document.createElement("strong");
        lab.className = "person person--" + p;
        lab.textContent = NAMES[p] + " is looking at:";
        row.appendChild(lab);
        const key = cam + "|" + it.frame_idx + "|" + p;
        const btns = [];
        for (const [value, text] of Object.entries(D.targets)) {
          const b = document.createElement("button");
          b.type = "button";
          b.textContent = text;
          const chosen = Boolean(state[key] && state[key].target === value);
          b.classList.toggle("chosen", chosen);
          b.setAttribute("aria-pressed", chosen ? "true" : "false");
          b.onclick = () => {
            state[key] = state[key] || {};
            state[key].target = value;
            btns.forEach((x) => {
              x.classList.toggle("chosen", x === b);
              x.setAttribute("aria-pressed", x === b ? "true" : "false");
            });
            refreshCard(card, cam, it.frame_idx);
            progress();
          };
          btns.push(b);
          row.appendChild(b);
        }
        const mark = document.createElement("button");
        mark.type = "button";
        mark.textContent = "Mark spot";
        mark.className = "mark-btn";
        mark.onclick = () => {
          if (armed) armed.button.classList.remove("armed");
          armed = { card, person: p, button: mark };
          mark.classList.add("armed");
        };
        row.appendChild(mark);
        card.appendChild(row);
      }
      refreshCard(card, cam, it.frame_idx);
      cards.appendChild(card);
    });
  }
  progress();

  document.getElementById("save").onclick = async () => {
    const labels = {};
    for (const [key, v] of Object.entries(state)) {
      if (!v.target) continue;
      const [cam, frame, person] = key.split("|");
      const it = D.items[cam].find((x) => String(x.frame_idx) === frame);
      (labels[cam] = labels[cam] || []).push({
        frame_idx: Number(frame),
        timestamp_s: it.timestamp_s,
        person,
        target: v.target,
        point: v.point || null,
      });
    }
    const status = document.getElementById("save-status");
    status.textContent = "Saving...";
    const res = await fetch(D.save_url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-csrf-token": D.csrf },
      body: JSON.stringify({ labeler: document.getElementById("labeler").value, labels }),
    });
    status.textContent = res.ok ? "Saved. Thank you!" : "Save failed: " + (await res.text());
  };
})();
