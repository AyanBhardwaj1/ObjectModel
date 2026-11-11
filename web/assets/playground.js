// Playground: runs /api/reconstruct and renders the result. All text goes in via textContent.
(function () {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const CDN = 'https://cdn.jsdelivr.net/npm/three@0.128.0/';
  let mode = 'demo';
  let lastFiles = {};

  function selectTab(which) {
    mode = which;
    for (const t of ['demo', 'upload']) {
      $('tab-' + t).setAttribute('aria-selected', String(t === which));
      $('pane-' + t).hidden = t !== which;
    }
  }
  $('tab-demo').onclick = () => selectTab('demo');
  $('tab-upload').onclick = () => selectTab('upload');
  const showNoise = () => { $('noise-out').textContent = (+$('noise').value).toFixed(1) + ' mm'; };
  const showSway = () => { const v = +$('sway').value; $('sway-out').textContent = v ? v.toFixed(2) + ' rad' : 'off'; };
  $('noise').oninput = showNoise;
  $('sway').oninput = showSway;

  const fmtRange = (j) => j.type === 'prismatic'
    ? `${Math.round(j.lower * 1000)} to ${Math.round(j.upper * 1000)} mm`
    : `${(j.lower * 180 / Math.PI).toFixed(1)} to ${(j.upper * 180 / Math.PI).toFixed(1)}°`;

  // The API returns a standalone viewer; show it in embed mode with the site's own copy of three.js.
  function viewerDoc(html) {
    return html
      .replace(CDN + 'build/three.min.js', location.origin + '/vendor/three/three.min.js')
      .replace(CDN + 'examples/js/controls/OrbitControls.js', location.origin + '/vendor/three/OrbitControls.js')
      .replace('<body>', '<body class="embed">');
  }

  function render(data) {
    const s = data.summary;
    $('viewer').srcdoc = viewerDoc(data.viewer_html);
    const sum = $('summary'); sum.replaceChildren();
    const tile = (value, label, cls) => { const d = el('div'); d.append(el('b', cls, value), el('span', null, label)); sum.append(d); };
    tile(s.sim_validated ? 'Validated' : 'Needs review', 'MuJoCo verdict', s.sim_validated ? 'ok' : 'bad');
    tile(s.unexplained_mm == null ? 'n/a' : s.unexplained_mm.toFixed(2) + ' mm', 'error not explained by noise');
    tile(s.urdf_mjcf_dev_mm == null ? 'n/a' : s.urdf_mjcf_dev_mm.toFixed(2) + ' mm', 'URDF and MJCF difference');
    tile(data.seconds.toFixed(1) + ' s', `${s.links} links, ${s.joints.length} joints`);
    const tb = $('joints'); tb.replaceChildren();
    if (!s.joints.length) {
      const tr = el('tr'); const td = el('td', null, 'No joints were found. Make sure the parts move during the capture.'); td.colSpan = 6; tr.append(td); tb.append(tr);
    }
    for (const j of s.joints) {
      const tr = el('tr');
      tr.append(el('td', null, j.name), el('td', null, j.type), el('td', null, `${j.parent} to ${j.child}`),
        el('td', 'n', fmtRange(j)), el('td', 'n', (j.confidence * 100).toFixed(1) + '%'),
        el('td', 'n', j.axis_std_deg == null ? 'n/a' : '±' + j.axis_std_deg.toFixed(2) + '°'));
      tb.append(tr);
    }
    lastFiles = data.files;
    const dl = $('downloads'); dl.replaceChildren();
    for (const name of Object.keys(data.files)) {
      const b = el('button', 'btn btn-ghost btn-sm', 'Download ' + name); b.type = 'button';
      b.onclick = () => {
        const url = URL.createObjectURL(new Blob([lastFiles[name]], { type: 'text/plain' }));
        const a = el('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
      };
      dl.append(b);
    }
    const iss = $('issues'); iss.replaceChildren();
    [...s.errors.map((e) => 'Error: ' + e), ...s.warnings.map((w) => 'Warning: ' + w), ...s.notes.map((n) => 'Note: ' + n)]
      .forEach((t) => iss.append(el('li', null, t)));
  }

  async function run(ev) {
    if (ev) ev.preventDefault();
    const btn = $('run'); const status = $('status');
    let req;
    if (mode === 'demo') {
      req = () => fetch('/api/reconstruct', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ object: $('object').value, noise_mm: +$('noise').value, camera_motion: +$('sway').value, seed: +$('seed').value || 0 }) });
    } else {
      const f = $('file').files[0];
      if (!f) { status.textContent = 'Choose a .npz file first.'; $('file').focus(); return; }
      if (f.size > 4 * 1024 * 1024) { status.textContent = 'That file is over 4 MB. Keep fewer tracks or frames and try again.'; $('file').focus(); return; }
      const hint = encodeURIComponent($('hint').value.trim());
      req = () => fetch('/api/reconstruct' + (hint ? '?hint=' + hint : ''), { method: 'POST', headers: { 'content-type': 'application/octet-stream' }, body: f });
    }
    btn.setAttribute('aria-busy', 'true'); btn.disabled = true; btn.textContent = 'Reconstructing…';
    status.textContent = 'Fitting joints and running MuJoCo…';
    try {
      const res = await req();
      const data = await res.json().catch(() => ({ error: 'The server sent a response we could not read. Try again in a moment.' }));
      if (!res.ok) throw new Error(data.error || `The request failed (${res.status}). Try again in a moment.`);
      render(data);
      status.textContent = `Done in ${data.seconds.toFixed(1)} s. Download the files below.`;
    } catch (e) {
      status.textContent = /fetch/i.test(e.message) ? 'Could not reach the server. Check your connection and try again.' : e.message;
    } finally {
      btn.removeAttribute('aria-busy'); btn.disabled = false; btn.textContent = 'Reconstruct';
    }
  }
  $('form').addEventListener('submit', run);

  // Arriving from "Rebuild a … now" on the home page runs that object straight away.
  const wanted = new URLSearchParams(location.search).get('object');
  if (wanted && [...$('object').options].some((o) => o.value === wanted)) {
    $('object').value = wanted;
    run();
  } else {
    fetch('/samples/kitchen_cabinet.json').then((r) => r.json()).then(render).catch(() => {
      $('status').textContent = 'Press Reconstruct to run your first model.';
    });
  }
})();
