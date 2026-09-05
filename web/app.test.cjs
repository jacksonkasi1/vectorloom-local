const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(handler) {
  const elements = new Map();
  function element() {
    const classes = new Set(['hidden']);
    return { src: '', textContent: '', innerHTML: '', disabled: false,
      classList: { add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x) },
      handlers: {}, addEventListener(name, handler) { this.handlers[name] = handler; }, querySelectorAll: () => [],
      replaceChildren(...children) { this.children = children; } };
  }
  const context = vm.createContext({
    document: { querySelector: selector => {
      if (!elements.has(selector)) elements.set(selector, element());
      return elements.get(selector);
    }, createElement: element },
    fetch: async (url, options) => url === '/api/models'
      ? response({ models: ['1b', '8b'].map(id => ({ id, label: id, installed: true, total_bytes: 1 })), runtime_device: 'Test' })
      : handler(url, options),
    FormData, Blob, URL: { createObjectURL: () => 'blob:result', revokeObjectURL() {} },
    setInterval() {}, clearInterval() {}, clearTimeout() {},
    setTimeout: callback => { if (callback.name !== 'loadModels') queueMicrotask(callback); },
  });
  vm.runInContext(fs.readFileSync(__dirname + '/app.js', 'utf8'), context);
  return { context, elements };
}
function response(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}
const result = { svg: '<svg/>', width: 285, height: 177, elapsed_ms: 25, engine: 'VTracer', warning: null };

test('logo upload defaults to direct tracing and displays a completed job', async () => {
  const calls = [];
  const { context, elements } = setup(async (url, options) => {
    calls.push(url);
    if (options?.method === 'POST') {
      assert.equal(options.body.get('model'), 'trace');
      return response({ job_id: 'job-1' }, 202);
    }
    return response({ state: 'complete', result });
  });
  await context.process(new Blob(['test']));
  assert.deepEqual(calls, ['/api/vectorize/jobs', '/api/vectorize/jobs/job-1']);
  assert.equal(elements.get('#preview').src, 'blob:result');
  assert.equal(elements.get('#result-warning').textContent, '');
  assert.equal(elements.get('#result').classList.contains('hidden'), false);
  assert.equal(elements.get('#image-input').disabled, false);
});

test('both model choices are sent with the individual request', async () => {
  for (const model of ['1b', '8b']) {
    const { context } = setup(async (url, options) => {
      if (options?.method === 'POST') {
        assert.equal(options.body.get('model'), model);
        return response({ job_id: 'ai' }, 202);
      }
      return response({ state: 'complete', result });
    });
    context.selectModel(model);
    await context.process(new Blob(['test']));
  }
});

test('polling survives transient connection/server failures and pending jobs', async () => {
  let count = 0;
  const { context } = setup(async () => {
    count++;
    if (count === 1) throw new Error('temporary network failure');
    if (count === 2) return response({}, 503);
    if (count === 3) return response({ state: 'processing' });
    return response({ state: 'complete', result });
  });
  assert.equal((await context.waitForJob('test')).svg, result.svg);
  assert.equal(count, 4);
});

test('failed jobs show the reason and re-enable upload', async () => {
  const { context, elements } = setup(async (url, options) => options
    ? response({ job_id: 'failed' }, 202)
    : response({ state: 'failed', error: 'Invalid image' }));
  await context.process(new Blob(['test']));
  assert.equal(elements.get('#error').textContent, 'Invalid image');
  assert.equal(elements.get('#image-input').disabled, false);
});

test('repeated polling errors stop instead of looping forever', async () => {
  const { context } = setup(async () => { throw new Error('offline'); });
  await assert.rejects(context.waitForJob('test'), /Connection lost/);
});

test('Mac download sends SVG text to the native save panel', async () => {
  const { context, elements } = setup(async (url, options) => options
    ? response({ job_id: 'native' }, 202) : response({ state: 'complete', result }));
  let saved;
  let prevented = false;
  context.webkit = { messageHandlers: { saveSVG: { postMessage: svg => { saved = svg; } } } };
  await context.process(new Blob(['test']));
  elements.get('#download').handlers.click({ preventDefault() { prevented = true; } });
  assert.equal(saved, result.svg);
  assert.equal(prevented, true);
});

test('manual model downloads appear only when local administration is enabled', () => {
  const { context, elements } = setup(async () => response({}));
  const catalog = { models: [{ id: '1b', label: '1B', installed: false, total_bytes: 1 }] };
  context.renderModels(catalog);
  assert.equal(elements.get('#models').children[1].innerHTML.includes('data-download='), false);
  context.renderModels({ ...catalog, model_admin_enabled: true });
  assert.equal(elements.get('#models').children[1].innerHTML.includes('data-download="1b"'), true);
});
