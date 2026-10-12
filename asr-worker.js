// Vazheyar speech-to-text worker.
// Runs Whisper (transformers.js) inside the browser: the audio never leaves this device.
// The library is pinned to one version and checked against its SHA-384 hash before it runs.
const LIB = {
  url: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js',
  sri: 'sha384-sOua6i337lNLWtuIG44PBf5JXIvhoicFClCcI4XHDe/eHW3dwhQ3Y6t6wQjfRrkT'
};
let T = null, pipe = null, pipeKey = '';
self.postMessage({ type: 'ver', v: 'tf3.8.1' });

async function sha384(buf) {
  const h = new Uint8Array(await crypto.subtle.digest('SHA-384', buf));
  let s = ''; for (const b of h) s += String.fromCharCode(b);
  return 'sha384-' + btoa(s);
}
async function lib() {
  if (T) return T;
  const r = await fetch(LIB.url, { credentials: 'omit' });
  if (!r.ok) throw new Error('network: library ' + r.status);
  const buf = await r.arrayBuffer();
  if (await sha384(buf) !== LIB.sri) throw new Error('integrity check failed');
  T = await import(URL.createObjectURL(new Blob([buf], { type: 'text/javascript' })));
  T.env.allowLocalModels = false;
  return T;
}
// Engine settings, tried in order. Each is checked with a one-second test run before real work starts,
// so a setting that loads but cannot run is skipped too.
async function plans(model) {
  let ad = null;
  try { ad = self.navigator.gpu ? await self.navigator.gpu.requestAdapter() : null; } catch (e) {}
  const f16 = !!(ad && ad.features && ad.features.has('shader-f16'));
  const out = [];
  if (ad && f16) out.push({ name: 'gpu-f16', device: 'webgpu', dtype: { encoder_model: 'fp16', decoder_model_merged: 'q4' } });
  else if (ad && /base/.test(model)) out.push({ name: 'gpu-f32', device: 'webgpu', dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } });
  out.push({ name: 'cpu-q8', device: 'wasm', dtype: 'q8' });
  out.push({ name: 'cpu-q8-basic', device: 'wasm', dtype: 'q8', session_options: { graphOptimizationLevel: 'basic' } });
  return out;
}
async function getPipe(model, post) {
  if (pipe && pipeKey === model) return pipe;
  const t = await lib();
  const progress_callback = p => {
    if (p && p.status === 'progress') post({ type: 'load', file: p.file, loaded: p.loaded, total: p.total });
  };
  const errors = [];
  for (const o of await plans(model)) {
    try {
      const opts = { progress_callback, device: o.device, dtype: o.dtype };
      if (o.session_options) opts.session_options = o.session_options;
      const p = await t.pipeline('automatic-speech-recognition', model, opts);
      await p(new Float32Array(16000), { language: 'dutch', task: 'transcribe' }); // test run
      pipe = p; pipeKey = model; post({ type: 'device', device: o.device, plan: o.name });
      return pipe;
    } catch (e) {
      const msg = String(e && e.message || e);
      errors.push(o.name + ': ' + msg.slice(0, 160));
      post({ type: 'note', msg: o.name + ' failed' });
      if (/integrity|network: library|Could not locate file: "[^"]*config\.json/i.test(msg)) break; // retrying other settings will not help
    }
  }
  throw new Error(errors.join(' | '));
}
self.onmessage = async e => {
  const m = e.data || {};
  const post = x => self.postMessage(x);
  if (m.type !== 'run') return;
  try {
    const p = await getPipe(m.model, post);
    const a = new Float32Array(m.audio);
    const n = m.wins.length;
    for (let i = 0; i < n; i++) {
      const [s, e2] = m.wins[i];
      const out = await p(a.subarray(s, e2), { language: m.lang, task: 'transcribe', return_timestamps: true });
      const off = s / 16000, end = e2 / 16000;
      const chunks = out && Array.isArray(out.chunks) && out.chunks.length ? out.chunks : [{ timestamp: [0, null], text: out && out.text }];
      const segs = chunks.map(c => {
        const ts = c.timestamp || [0, null];
        const st = Math.min(end, off + (ts[0] || 0));
        const en = ts[1] == null ? end : Math.min(end, off + ts[1]);
        return { s: Math.round(st * 100) / 100, e: Math.round(Math.max(st + 0.2, en) * 100) / 100, t: String(c.text || '').replace(/\s+/g, ' ').trim() };
      }).filter(x => x.t && !/^[\[(].*[\])]$/.test(x.t));
      post({ type: 'part', i, n, segs });
    }
    post({ type: 'done' });
  } catch (err) {
    post({ type: 'error', msg: String(err && err.message || err) });
  }
};
