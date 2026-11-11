// Landing page: the live hero fragment, capability switches and the gallery plate.
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  // Hero: object tabs drive the embedded viewer; the viewer reports every frame back.
  const frame = $('#hero-viewer');
  const readout = $('#readout');
  const fmt = (j) => j.type === 'prismatic' ? `${Math.round(j.value * 1000)} mm` : `${(j.value * 180 / Math.PI).toFixed(1)}°`;
  $$('.fragment-tabs [role=tab]').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.fragment-tabs [role=tab]').forEach((t) => t.setAttribute('aria-selected', String(t === tab)));
      try { frame.contentWindow.location.hash = tab.dataset.object; } catch (e) { frame.src = '/gallery?embed=1#' + tab.dataset.object; }
      frame.title = tab.textContent + ' replaying its capture in 3D';
    });
  });
  let last = 0;
  window.addEventListener('message', (ev) => {
    if (ev.source !== frame.contentWindow || !ev.data || ev.data.type !== 'om-frame') return;
    frame.previousElementSibling.hidden = true; // the live view is running; drop the still
    const now = performance.now();
    if (now - last < 120) return; // readable cadence
    last = now;
    const cells = [el('div', ev.data.passed ? 'ok' : '')];
    cells[0].append(el('b', null, ev.data.passed ? 'Validated' : 'Review'), el('span', null, 'MuJoCo verdict'));
    ev.data.joints.slice(0, 4).forEach((j) => { const d = el('div'); d.append(el('b', 'num', fmt(j)), el('span', null, `${j.name}, ${j.type}`)); cells.push(d); });
    readout.replaceChildren(...cells);
  });

  // Pause controls for the two moving pictures; reduced motion starts them paused.
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const heroPause = $('#hero-pause');
  const setHero = (paused) => {
    heroPause.setAttribute('aria-pressed', String(paused));
    heroPause.textContent = paused ? 'Play' : 'Pause';
    try { frame.contentWindow.postMessage({ type: 'om-control', play: !paused }, '*'); } catch (e) { /* frame not ready */ }
  };
  heroPause.addEventListener('click', () => setHero(heroPause.getAttribute('aria-pressed') !== 'true'));
  frame.addEventListener('load', () => { if (reduce) setHero(true); });
  const clip = $('#clip');
  const clipPause = $('#clip-pause');
  const setClip = (paused) => {
    if (paused) clip.pause(); else clip.play().catch(() => { /* autoplay refused by the browser: stays paused, button shows Play */ clipPause.textContent = 'Play'; });
    clipPause.setAttribute('aria-pressed', String(paused));
    clipPause.textContent = paused ? 'Play' : 'Pause';
    clipPause.setAttribute('aria-label', (paused ? 'Play' : 'Pause') + ' the replay clip');
  };
  clipPause.addEventListener('click', () => setClip(!clip.paused ? true : false));
  if (reduce) setClip(true);

  // Capability switches: Space and Enter work because they are real buttons.
  $$('#bands .switch').forEach((sw) => {
    sw.addEventListener('click', () => {
      const on = sw.getAttribute('aria-checked') !== 'true';
      sw.setAttribute('aria-checked', String(on));
      sw.closest('.band').dataset.on = String(on);
    });
  });

  // Gallery plate tabs.
  const plates = {
    dresser: { img: '/img/square-dresser.webp', alt: 'The reconstructed dresser with three drawers and a door', title: 'Five parts, one replay',
      desc: 'Three drawers and a door, each found on its own and each with its own joint. Axis error 0.51° on average, and every drawer slides out of the body rather than into it.' },
    desk_lamp: { img: '/img/square-desk-lamp.webp', alt: 'The reconstructed desk lamp: base, lower arm and upper arm', title: 'A chain, not a guess',
      desc: 'The upper arm moves with the lower one, so it hangs off the elbow, not the base. ObjectModel finds that because two hinges in a row are never one hinge.' },
    fridge: { img: '/img/square-fridge.webp', alt: 'The reconstructed fridge with its two doors on mirrored hinges', title: 'Mirrored doors, both right',
      desc: 'Left door hinged on the left, right door on the right, both vertical to within 0.04°. The joint sweep confirms they never collide.' },
  };
  $$('.plate-tabs [role=tab]').forEach((tab) => {
    tab.addEventListener('click', () => {
      const p = plates[tab.dataset.plate];
      $$('.plate-tabs [role=tab]').forEach((t) => t.setAttribute('aria-selected', String(t === tab)));
      const img = $('#plate-img'); img.src = p.img; img.alt = p.alt;
      $('#plate-title').textContent = p.title;
      $('#plate-desc').textContent = p.desc;
      $('#plate-open').href = '/gallery#' + tab.dataset.plate;
    });
  });
})();
