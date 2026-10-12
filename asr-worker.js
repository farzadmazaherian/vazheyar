// Vazheyar speech-to-text worker.
// Runs Whisper (transformers.js) inside the browser: the audio never leaves this device.
// The library is pinned to one version and checked against its SHA-384 hash before it runs.
const LIB = {
  url: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js',
  sri: 'sha384-3TA7bWonh/FGdhkvnr9EMFCoFqikJ5aJQEV8RlUgOmBFIF9z3W+h67aRdQPRxN9d'
};
let T = null, pipe = null, pipeKey = '';

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
async function getPipe(model, post) {
  if (pipe && pipeKey === model) return pipe;
  const t = await lib();
  const progress_callback = p => {
    if (p && p.status === 'progress') post({ type: 'load', file: p.file, loaded: p.loaded, total: p.total });
  };
  // Graphics card (WebGPU) only when it supports half precision; otherwise the processor with 8-bit weights.
  // small: ~410 MB on WebGPU, ~250 MB on the processor; base: ~165 MB / ~77 MB. Downloaded once, then cached.
  let f16 = false;
  try { const ad = self.navigator.gpu && await self.navigator.gpu.requestAdapter(); f16 = !!(ad && ad.features && ad.features.has('shader-f16')); } catch (e) {}
  const tries = f16
    ? [{ device: 'webgpu', dtype: { encoder_model: 'fp16', decoder_model_merged: 'q4' } }, { device: 'wasm', dtype: 'q8' }]
    : [{ device: 'wasm', dtype: 'q8' }];
  let last;
  for (const o of tries) {
    try {
      pipe = await t.pipeline('automatic-speech-recognition', model, Object.assign({ progress_callback }, o));
      pipeKey = model; post({ type: 'device', device: o.device });
      return pipe;
    } catch (e) { last = e; }
  }
  throw last;
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
