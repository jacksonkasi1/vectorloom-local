const input = document.querySelector('#image-input');
const dropzone = document.querySelector('#dropzone');
const progress = document.querySelector('#progress');
const progressText = document.querySelector('#progress-text');
const result = document.querySelector('#result');
const preview = document.querySelector('#preview');
const download = document.querySelector('#download');
const error = document.querySelector('#error');
const meta = document.querySelector('#result-meta');
const runtime = document.querySelector('#runtime');
const models = document.querySelector('#models');
const device = document.querySelector('#device');
const hardwareNote = document.querySelector('#hardware-note');
const resultWarning = document.querySelector('#result-warning');

let modelPoll;
let selectedModel = 'trace';
let processing = false;
let currentSVG;
download.addEventListener('click', event => {
  const nativeSave = globalThis.webkit?.messageHandlers?.saveSVG;
  if (nativeSave && currentSVG) {
    event.preventDefault();
    nativeSave.postMessage(currentSVG);
  }
});
loadModels();

input.addEventListener('change', () => input.files[0] && process(input.files[0]));
['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault(); dropzone.classList.add('over');
}));
['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault(); dropzone.classList.remove('over');
}));
dropzone.addEventListener('drop', event => event.dataTransfer.files[0] && process(event.dataTransfer.files[0]));

async function process(file) {
  if (processing) return;
  processing = true;
  input.disabled = true;
  error.classList.add('hidden'); result.classList.add('hidden'); progress.classList.remove('hidden');
  const startedAt = Date.now();
  progressText.textContent = 'Building vectors… 0h 0m 0s';
  const elapsedTimer = setInterval(() => {
    progressText.textContent = `Building vectors… ${formatDuration(Date.now() - startedAt)}`;
  }, 1000);
  const form = new FormData();
  form.append('image', file);
  form.append('model', selectedModel || '8b');
  try {
    const response = await fetch('/api/vectorize/jobs', { method: 'POST', body: form });
    const accepted = await response.json();
    if (!response.ok) throw new Error(accepted.error || 'Could not start conversion.');
    const payload = await waitForJob(accepted.job_id);
    currentSVG = payload.svg;
    const blob = new Blob([payload.svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
    preview.src = url;
    download.href = url;
    meta.textContent = `${payload.width} × ${payload.height} · ${formatDuration(payload.elapsed_ms)} · ${payload.engine}`;
    resultWarning.textContent = payload.warning || '';
    result.classList.remove('hidden');
  } catch (err) {
    error.textContent = err.message; error.classList.remove('hidden');
  } finally {
    processing = false;
    input.disabled = false;
    clearInterval(elapsedTimer);
    progress.classList.add('hidden');
  }
}

async function waitForJob(jobId) {
  let failures = 0;
  while (true) {
    let response;
    try {
      response = await fetch(`/api/vectorize/jobs/${encodeURIComponent(jobId)}`);
    } catch (err) {
      if (++failures > 5) throw new Error('Connection lost while checking conversion.');
      await new Promise(resolve => setTimeout(resolve, 2000));
      continue;
    }
    if (response.status >= 500 && ++failures <= 5) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      continue;
    }
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Could not check conversion.');
    failures = 0;
    if (job.state === 'complete') return job.result;
    if (job.state === 'failed') throw new Error(job.error || 'Conversion failed.');
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
}

async function loadModels() {
  try {
    const response = await fetch('/api/models');
    const catalog = await response.json();
    renderModels(catalog);
    clearTimeout(modelPoll);
    modelPoll = setTimeout(loadModels, catalog.models.some(model => model.phase === 'downloading') ? 700 : 4000);
  } catch (_) {
    device.textContent = 'Runtime unavailable';
  }
}

function renderModels(catalog) {
  device.textContent = catalog.runtime_device;
  hardwareNote.textContent = catalog.hardware_note;
  const selected = catalog.models.find(model => model.id === selectedModel);
  selectedModel ||= selected?.id || '8b';
  runtime.textContent = selectedModel === 'trace' ? 'Direct tracing · preserves source shapes' : selected?.installed
    ? `${selected.label} · ${catalog.runtime_device}`
    : `${selected?.label || 'Model'} selected · automatic tracer available`;
  const traceCard = document.createElement('article');
  traceCard.className = `model-card${selectedModel === 'trace' ? ' selected' : ''}`;
  traceCard.innerHTML = `<div class="model-name"><strong>Direct tracing</strong><span>For logos</span></div>
    <p>Preserves lettering and source shapes. No AI model loading.</p>
    <div class="model-actions"><button data-select="trace" ${selectedModel === 'trace' ? 'disabled' : ''}>${selectedModel === 'trace' ? 'Selected' : 'Use tracing'}</button></div>`;
  models.replaceChildren(traceCard, ...catalog.models.map(model => {
    const card = document.createElement('article');
    const isSelected = model.id === selectedModel;
    card.className = `model-card${isSelected ? ' selected' : ''}`;
    const percent = Math.min(100, Math.round((model.downloaded_bytes / model.total_bytes) * 100));
    card.innerHTML = `
      <div class="model-name"><strong>${model.label}</strong><span>AI generation</span></div>
      <p>${formatBytes(model.total_bytes)} checkpoint · ${model.installed ? 'Installed' : model.phase === 'downloading' ? `${percent}% downloaded` : 'Not downloaded'}. May change lettering or fine details.</p>
      <div class="model-progress"><span style="width:${model.installed ? 100 : percent}%"></span></div>
      <div class="model-actions">
        <button data-select="${model.id}" ${isSelected || !model.installed ? 'disabled' : ''}>${isSelected ? 'Selected' : model.installed ? 'Use model' : 'Preparing…'}</button>
        ${catalog.model_admin_enabled && !model.installed ? `<button data-download="${model.id}" ${model.phase === 'downloading' ? 'disabled' : ''}>${model.phase === 'downloading' ? 'Downloading…' : 'Download model'}</button>` : ''}
      </div>
      ${model.message ? `<small>${escapeHtml(model.message)}</small>` : ''}`;
    return card;
  }));
  models.querySelectorAll('[data-select]').forEach(button => button.addEventListener('click', () => selectModel(button.dataset.select)));
  models.querySelectorAll('[data-download]').forEach(button => button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      const response = await fetch(`/api/models/${button.dataset.download}/download`, { method: 'POST' });
      if (!response.ok) throw new Error((await response.json()).error || 'Could not start download.');
      await loadModels();
    } catch (err) {
      error.textContent = err.message; error.classList.remove('hidden'); button.disabled = false;
    }
  }));
}

function selectModel(model) {
  selectedModel = model;
  loadModels();
}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.max(0, Math.round(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${hours}h ${minutes}m ${seconds}s`;
}

function escapeHtml(value) {
  const node = document.createElement('span'); node.textContent = value; return node.innerHTML;
}
