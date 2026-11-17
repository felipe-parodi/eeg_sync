// Upload page: chunked video upload, drag-and-drop, logout.
// CHUNK_SIZE and CSRF_TOKEN are defined by an inline config script in index.html.
const form = document.getElementById("upload-form");
const logoutButton = document.getElementById("logout-button");
const statusEl = document.getElementById("upload-status");
const progressEl = document.getElementById("upload-progress");
const dropZones = document.querySelectorAll(".file-drop-zone");
let uploadInProgress = false;

function setStatus(message) {
  statusEl.textContent = message;
}

function formatBytes(bytes) {
  const gib = bytes / (1024 * 1024 * 1024);
  if (gib >= 1) {
    return `${gib.toFixed(2)} GB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateSelectedFile(input) {
  const label = document.getElementById(`${input.id}_selected`);
  if (!label) {
    return;
  }
  label.textContent = input.files.length ? input.files[0].name : "No file selected";
}

logoutButton.addEventListener("click", async () => {
  await fetch("/logout", {
    method: "POST",
    headers: {"X-CSRF-Token": CSRF_TOKEN},
    credentials: "same-origin"
  });
  window.location.href = "/login";
});

dropZones.forEach((zone) => {
  const input = document.getElementById(zone.dataset.fileInput);
  input.addEventListener("change", () => updateSelectedFile(input));

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => {
    zone.classList.remove("dragover");
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragover");
    if (!event.dataTransfer.files.length) {
      return;
    }
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    input.files = transfer.files;
    updateSelectedFile(input);
  });
});

async function postFormData(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
    headers: {"X-CSRF-Token": CSRF_TOKEN},
    credentials: "same-origin"
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return await response.json();
}

function postChunkFormData(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.withCredentials = true;
    request.setRequestHeader("X-CSRF-Token", CSRF_TOKEN);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded, event.total);
      }
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(JSON.parse(request.responseText));
        return;
      }
      reject(new Error(request.responseText || `Request failed: ${request.status}`));
    };
    request.onerror = () => reject(new Error("Network error during chunk upload."));
    request.send(formData);
  });
}

function updateUploadProgress(cameraId, cameraLabel, chunkNumber, totalChunks, visibleCameraBytes, uploadState) {
  uploadState[cameraId] = visibleCameraBytes;
  const visibleBytes = uploadState.camera_a + uploadState.camera_b;
  progressEl.value = Math.round((visibleBytes / uploadState.totalBytes) * 100);
  setStatus(
    `${cameraLabel}: uploading chunk ${chunkNumber} of ${totalChunks} ` +
    `(${formatBytes(visibleBytes)} of ${formatBytes(uploadState.totalBytes)} total). Keep this tab open.`
  );
}

async function uploadCamera(jobId, cameraId, file, uploadState) {
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
  const cameraLabel = cameraId === "camera_a" ? "Video A" : "Video B";
  let uploadedCameraBytes = 0;
  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * CHUNK_SIZE;
    const end = Math.min(file.size, start + CHUNK_SIZE);
    const chunk = file.slice(start, end);
    const formData = new FormData();
    formData.append("camera_id", cameraId);
    formData.append("chunk_index", String(index));
    formData.append("total_chunks", String(totalChunks));
    formData.append("chunk", chunk, file.name);
    setStatus(`${cameraLabel}: uploading chunk ${index + 1} of ${totalChunks}. Keep this tab open.`);
    await postChunkFormData(`/uploads/${jobId}/chunk`, formData, (loaded) => {
      updateUploadProgress(
        cameraId,
        cameraLabel,
        index + 1,
        totalChunks,
        uploadedCameraBytes + loaded,
        uploadState
      );
    });
    uploadedCameraBytes += chunk.size;
    updateUploadProgress(
      cameraId,
      cameraLabel,
      index + 1,
      totalChunks,
      uploadedCameraBytes,
      uploadState
    );
  }
}

window.addEventListener("beforeunload", (event) => {
  if (!uploadInProgress) {
    return;
  }
  event.preventDefault();
  event.returnValue = "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitButton = form.querySelector("button");
  submitButton.disabled = true;
  submitButton.textContent = "Uploading...";
  uploadInProgress = true;
  document.title = "Uploading videos";
  progressEl.style.display = "block";
  progressEl.value = 0;

  try {
    const videoA = document.getElementById("video_a").files[0];
    const videoB = document.getElementById("video_b").files[0];
    const blocksA = document.getElementById("blocks_text_a").value.trim();
    const blocksB = document.getElementById("blocks_text_b").value.trim();
    if (!videoA && !videoB) {
      throw new Error("Upload at least one GoPro video.");
    }
    if (videoA && !blocksA) {
      throw new Error("Video A analysis blocks are required when Video A is uploaded.");
    }
    if (videoB && !blocksB) {
      throw new Error("Video B analysis blocks are required when Video B is uploaded.");
    }

    const initData = new FormData(form);
    initData.append("video_a_name", videoA ? videoA.name : "");
    initData.append("video_b_name", videoB ? videoB.name : "");
    initData.append("video_a_size", videoA ? String(videoA.size) : "0");
    initData.append("video_b_size", videoB ? String(videoB.size) : "0");
    initData.delete("video_a");
    initData.delete("video_b");

    setStatus("Creating upload job");
    const init = await postFormData("/uploads/init", initData);
    const totalBytes = (videoA ? videoA.size : 0) + (videoB ? videoB.size : 0);
    const uploadState = { camera_a: 0, camera_b: 0, totalBytes };
    const uploadTasks = [];
    if (videoA) {
      uploadTasks.push(uploadCamera(init.job_id, "camera_a", videoA, uploadState));
    }
    if (videoB) {
      uploadTasks.push(uploadCamera(init.job_id, "camera_b", videoB, uploadState));
    }
    await Promise.all(uploadTasks);

    setStatus("Upload complete. Asking server to finalize chunks and start processing.");
    document.title = "Processing queued";
    const finish = await postFormData(`/uploads/${init.job_id}/finish`, new FormData());
    uploadInProgress = false;
    window.location.href = finish.status_url;
  } catch (error) {
    setStatus(`Upload failed: ${error.message}`);
    submitButton.disabled = false;
    submitButton.textContent = "Submit Processing Job";
    uploadInProgress = false;
    document.title = "Submit Job";
  }
});
