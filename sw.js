// Vazheyar service worker: keeps the app working offline.
const CACHE = 'vazheyar-v16';
const SHELL = ['./', './index.html', './packs.js?v=16', './asr-worker.js?v=16', './manifest.webmanifest', './icon-192.png', './icon-512.png', './icon-maskable-512.png', './apple-touch-icon.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL.map(u => new Request(u, { cache: 'reload' })))).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE && k.startsWith('vazheyar-')).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.hostname === 'api.github.com') return; // sync data is never cached

  // Pages: try the network first so updates arrive, fall back to the cached copy offline.
  if (req.mode === 'navigate') {
    e.respondWith(
      fetch(req, { cache: 'no-cache' }).then(res => {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put('./index.html', copy));
        return res;
      }).catch(() => caches.match('./index.html').then(r => r || caches.match('./')))
    );
    return;
  }

  // App files, fonts and the small pinned helper libraries: serve from cache, refresh in the background.
  // Speech models and other large downloads are cached by the libraries themselves, not here.
  const lib = url.hostname === 'cdn.jsdelivr.net' && /^\/npm\/(pdfjs-dist|tesseract\.js|qrcode-generator)@[\d.]+\/.+\.(m?js)$/.test(url.pathname);
  const cacheable = (url.origin === location.origin && !url.search.includes('check=')) || lib || /(^|\.)fonts\.(googleapis|gstatic)\.com$/.test(url.hostname);
  if (!cacheable) return;
  e.respondWith(
    caches.match(req).then(hit => {
      const net = fetch(req).then(res => {
        if (res.ok || res.type === 'opaque') {
          const copy = res.clone();
          caches.open(CACHE).then(c => c.put(req, copy));
        }
        return res;
      }).catch(() => hit);
      return hit || net;
    })
  );
});
