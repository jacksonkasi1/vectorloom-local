const input = document.querySelector('#image-input');
const dropzone = document.querySelector('#dropzone');
const progress = document.querySelector('#progress');
const result = document.querySelector('#result');
const preview = document.querySelector('#preview');
const download = document.querySelector('#download');
const error = document.querySelector('#error');
const meta = document.querySelector('#result-meta');
const runtime = document.querySelector('#runtime');

fetch('/api/health').then(r => r.json()).then(({ status }) => {
  runtime.textContent = `${status.requested_model} · ${status.fallback_engine}`;
}).catch(() => { runtime.textContent = 'Local runtime ready'; });

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
  const form = new FormData(); form.append('image', file);
  try {
    const response = await fetch('/api/vectorize', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Vectorization failed.');
    const blob = new Blob([payload.svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
    preview.src = url; download.href = url;
    meta.textContent = `${payload.width} × ${payload.height} · ${payload.elapsed_ms} ms · ${payload.engine}`;
    result.classList.remove('hidden');
  } catch (err) {
    error.textContent = err.message; error.classList.remove('hidden');
  } finally { progress.classList.add('hidden'); }
}
