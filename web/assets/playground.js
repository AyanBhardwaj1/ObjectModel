// Playground: runs /api/reconstruct and renders the result. All text goes in via textContent.
(function () {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
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
  $('noise').oninput = () => { $('noise-out').textContent = (+$('noise').value).toFixed(1) + ' mm'; };
  $('sway').oninput = () => { const v = +$('sway').value; $('sway-out').textContent = v ? v.toFixed(2) + ' rad' : 'off'; };

  const fmtRange = (j) => j.type === 'prismatic'
    ? `${(j.lower * 1000).toFixed(0)} … ${(j.upper * 1000).toFixed(0)} mm`
    : `${(j.lower * 180 / Math.PI).toFixed(1)} … ${(j.upper * 180 / Math.PI).toFixed(1)}°`;

  function render(data) {
    const s = data.summary;
    $('viewer').srcdoc = data.viewer_html;
    const sum = $('summary'); sum.replaceChildren();
    const tile = (value, label, cls) => { const d = el('div'); const b = el('b', cls, value); d.append(b, el('span', null, label)); sum.append(d); };
    tile(s.sim_validated ? 'Validated' : 'Review', 'MuJoCo verdict', s.sim_validated ? 'ok' : 'bad');
    tile(s.unexplained_mm == null ? '—' : s.unexplained_mm.toFixed(2) + ' mm', 'error not explained by noise');
    tile(s.urdf_mjcf_dev_mm == null ? '—' : s.urdf_mjcf_dev_mm.toFixed(2) + ' mm', 'URDF vs MJCF difference');
    tile(data.seconds.toFixed(1) + ' s', `${s.links} links · ${s.joints.length} joints`);
    const tb = $('joints'); tb.replaceChildren();
    for (const j of s.joints) {
      const tr = el('tr');
      tr.append(el('td', null, j.name), el('td', null, j.type), el('td', null, `${j.parent} → ${j.child}`),
        el('td', 'n', fmtRange(j)), el('td', 'n', (j.confidence * 100).toFixed(1) + '%'),
        el('td', 'n', j.axis_std_deg == null ? 'n/a' : j.axis_std_deg.toFixed(2) + '°'));
      tb.append(tr);
    }
    lastFiles = data.files;
    const dl = $('downloads'); dl.replaceChildren();
    for (const name of Object.keys(data.files)) {
      const b = el('button', 'btn btn-ghost btn-sm', '↓ ' + name); b.type = 'button';
      b.onclick = () => {
        const url = URL.createObjectURL(new Blob([lastFiles[name]], { type: 'text/plain' }));
        const a = el('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
      };
      dl.append(b);
    }
    const iss = $('issues'); iss.replaceChildren();
    [...s.errors.map((e) => '✗ ' + e), ...s.warnings.map((w) => '⚠ ' + w), ...s.notes.map((n) => '• ' + n)].forEach((t) => iss.append(el('li', null, t)));
  }

  async function run(ev) {
    ev.preventDefault();
    const btn = $('run'); const status = $('status');
    let req;
    if (mode === 'demo') {
      req = fetch('/api/reconstruct', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ object: $('object').value, noise_mm: +$('noise').value, camera_motion: +$('sway').value, seed: +$('seed').value || 0 }) });
    } else {
      const f = $('file').files[0];
      if (!f) { status.textContent = 'Choose a .npz file first.'; return; }
      if (f.size > 4 * 1024 * 1024) { status.textContent = 'That file is over 4 MB. Subsample tracks or frames and try again.'; return; }
      const hint = encodeURIComponent($('hint').value.trim());
      req = fetch('/api/reconstruct' + (hint ? '?hint=' + hint : ''), { method: 'POST', headers: { 'content-type': 'application/octet-stream' }, body: f });
    }
    btn.setAttribute('aria-busy', 'true'); btn.disabled = true; btn.textContent = 'Reconstructing…';
    status.textContent = 'Fitting joints and running MuJoCo. This takes a few seconds.';
    try {
      const res = await req;
      const data = await res.json().catch(() => ({ error: 'The server sent an unreadable response.' }));
      if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
      render(data);
      status.textContent = `Done in ${data.seconds.toFixed(1)} s.`;
    } catch (e) {
      status.textContent = e.message.startsWith('Failed to fetch') ? 'Could not reach the server. Check your connection and try again.' : e.message;
    } finally {
      btn.removeAttribute('aria-busy'); btn.disabled = false; btn.textContent = 'Reconstruct';
    }
  }
  $('form').addEventListener('submit', run);

  fetch('/samples/kitchen_cabinet.json').then((r) => r.json()).then(render).catch(() => {
    $('status').textContent = 'Press Reconstruct to run your first model.';
  });
})();
