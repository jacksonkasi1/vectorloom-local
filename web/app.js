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
  error.classList.add('hidden'); result.classList.add('hidden'); progress.classList.remove('hidden');
  const startedAt = Date.now();
  progressText.textContent = 'Building vectors… 0h 0m 0s';
  const elapsedTimer = setInterval(() => {
    progressText.textContent = `Building vectors… ${formatDuration(Date.now() - startedAt)}`;
  }, 1000);
  const form = new FormData(); form.append('image', file);
  try {
    const response = await fetch('/api/vectorize', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Vectorization failed.');
    const blob = new Blob([payload.svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
    preview.src = url;
    download.href = `/api/download?t=${Date.now()}`;
    meta.textContent = `${payload.width} × ${payload.height} · ${formatDuration(payload.elapsed_ms)} · ${payload.engine}`;
    resultWarning.textContent = payload.warning || '';
    result.classList.remove('hidden');
  } catch (err) {
    error.textContent = err.message; error.classList.remove('hidden');
  } finally {
    clearInterval(elapsedTimer);
    progress.classList.add('hidden');
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
  const selected = catalog.models.find(model => model.selected);
  runtime.textContent = selected?.installed
    ? `${selected.label} · ${catalog.runtime_device}`
    : `${selected?.label || 'Model'} selected · automatic tracer available`;
  models.replaceChildren(...catalog.models.map(model => {
    const card = document.createElement('article');
    card.className = `model-card${model.selected ? ' selected' : ''}`;
    const percent = Math.min(100, Math.round((model.downloaded_bytes / model.total_bytes) * 100));
    card.innerHTML = `
      <div class="model-name"><strong>${model.label}</strong><span>${model.id === '8b' ? 'Best quality' : 'Faster'}</span></div>
      <p>${formatBytes(model.total_bytes)} checkpoint · ${model.installed ? 'Ready locally' : model.phase === 'downloading' ? `${percent}% downloaded` : 'Not downloaded'}</p>
      <div class="model-progress"><span style="width:${model.installed ? 100 : percent}%"></span></div>
      <div class="model-actions">
        <button data-select="${model.id}" ${model.selected ? 'disabled' : ''}>${model.selected ? 'Selected' : 'Use model'}</button>
        ${model.installed ? '' : `<button class="secondary" data-download="${model.id}" ${model.phase === 'downloading' ? 'disabled' : ''}>${model.phase === 'downloading' ? 'Downloading…' : 'Download'}</button>`}
      </div>
      ${model.message ? `<small>${escapeHtml(model.message)}</small>` : ''}`;
    return card;
  }));
  models.querySelectorAll('[data-select]').forEach(button => button.addEventListener('click', () => selectModel(button.dataset.select)));
  models.querySelectorAll('[data-download]').forEach(button => button.addEventListener('click', () => downloadModel(button.dataset.download)));
}

async function selectModel(model) {
  const response = await fetch('/api/models/select', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model })
  });
  if (!response.ok) throw new Error(`Could not select StarVector ${model.toUpperCase()}.`);
  await loadModels();
}

async function downloadModel(model) {
  const response = await fetch(`/api/models/${model}/download`, { method: 'POST' });
  if (!response.ok) {
    const payload = await response.json();
    error.textContent = payload.error; error.classList.remove('hidden');
  }
  await loadModels();
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
