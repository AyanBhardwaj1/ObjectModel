// Plan map: a top-down drawing of the plan `objectmodel plan` exports for each part.
// Everything here is read from /plans/plans.json; the part's motion uses the twin's own joint.
(() => {
  const root = document.querySelector('[data-plan-map]');
  if (!root) return;
  const svg = root.querySelector('svg');
  const tabs = root.querySelector('[data-plan-tabs]');
  const readout = root.querySelector('[data-plan-readout]');
  const code = root.querySelector('[data-plan-json]');
  const pauseBtn = root.querySelector('[data-plan-pause]');
  const NS = 'http://www.w3.org/2000/svg';
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const NAMES = { kitchen_cabinet: 'Cabinet', fridge: 'Fridge', dresser: 'Dresser', drawer_cabinet: 'Drawer unit', laptop: 'Laptop' };

  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const mul = (a, s) => a.map((x) => x * s);
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a) => mul(a, 1 / Math.hypot(...a));
  // Rodrigues rotation of p about the line (origin o, axis u) by angle t
  const rotate = (p, o, u, t) => {
    const v = sub(p, o), c = Math.cos(t), s = Math.sin(t);
    return add(o, add(add(mul(v, c), mul(cross(u, v), s)), mul(u, dot(u, v) * (1 - c))));
  };
  const corners = (b) => {
    const out = [];
    for (const i of [-0.5, 0.5]) for (const j of [-0.5, 0.5]) for (const k of [-0.5, 0.5]) {
      out.push(add(b.center, add(add(mul(b.axes[0], i * b.size[0]), mul(b.axes[1], j * b.size[1])), mul(b.axes[2], k * b.size[2]))));
    }
    return out;
  };
  // convex hull of the top-down projection (monotone chain)
  const hull = (pts) => {
    const p = pts.map((q) => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const turn = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const half = (list) => { const h = []; for (const q of list) { while (h.length >= 2 && turn(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop(); h.push(q); } h.pop(); return h; };
    return half(p).concat(half(p.slice().reverse()));
  };
  const pts = (list) => list.map((q) => `${q[0].toFixed(4)},${(-q[1]).toFixed(4)}`).join(' ');
  const el = (name, attrs, parent) => { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).append(e); return e; };

  let data = null, current = null, raf = 0, paused = reduce, t0 = 0, phase = 0;

  function moved(task, p, q) {
    const j = task.joint, dq = q - j.q_closed;
    const u = unit(j.axis);
    return j.type === 'prismatic' ? add(p, mul(u, dq)) : rotate(p, j.origin, u, dq);
  }

  function draw(task) {
    svg.replaceChildren();
    const g = task.geometry, st = task.stance;
    const closed = corners(g.part_closed);
    const all = corners(g.parent).concat(closed, corners(g.part_open), task.path.map((w) => w.handle), [st.base_xyz]);
    const xs = all.map((q) => q[0]), ys = all.map((q) => q[1]);
    const pad = 0.32;
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
    const side = Math.max(x1 - x0, y1 - y0);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    svg.setAttribute('viewBox', `${cx - side / 2} ${-cy - side / 2} ${side} ${side}`);
    const s = side / 100; // one percent of the view, for strokes and labels

    // 10 cm grid
    const grid = el('g', { class: 'pm-grid' });
    for (let x = Math.ceil((cx - side / 2) * 10) / 10; x < cx + side / 2; x += 0.1) el('line', { x1: x, x2: x, y1: -cy - side / 2, y2: -cy + side / 2, 'stroke-width': s * 0.15 }, grid);
    for (let y = Math.ceil((cy - side / 2) * 10) / 10; y < cy + side / 2; y += 0.1) el('line', { y1: -y, y2: -y, x1: cx - side / 2, x2: cx + side / 2, 'stroke-width': s * 0.15 }, grid);

    el('polygon', { class: 'pm-body', points: pts(hull(corners(g.parent))) });
    el('polygon', { class: 'pm-closed', points: pts(hull(closed)), 'stroke-width': s * 0.5, 'stroke-dasharray': `${s * 1.5} ${s}` });
    const part = el('polygon', { class: 'pm-part', 'stroke-width': s * 0.5 });
    el('polyline', { class: 'pm-path', points: pts(task.path.map((w) => w.handle)), 'stroke-width': s * 0.6 });
    for (const w of task.path) el('circle', { class: 'pm-way', cx: w.handle[0], cy: -w.handle[1], r: s * 0.55 });
    if (task.joint.type !== 'prismatic') {
      const o = task.joint.origin;
      el('circle', { class: 'pm-hinge', cx: o[0], cy: -o[1], r: s * 1.3, 'stroke-width': s * 0.5 });
      label('hinge', [o[0], o[1]], [-s * 2.4, s * 1.1], s, 'end');
    }
    // approach direction at the handle
    const h = task.handle, n = task.approach_normal;
    const tip = add(h, mul(n, 0.14));
    el('line', { class: 'pm-approach', x1: h[0], y1: -h[1], x2: tip[0], y2: -tip[1], 'stroke-width': s * 0.5, 'stroke-dasharray': `${s} ${s * 0.8}` });

    // the robot, seen from above: torso, facing direction, working shoulder and its reach to the handle
    const b = st.base_xyz, yaw = st.base_yaw;
    const robot = el('g', { class: 'pm-robot', transform: `translate(${b[0]} ${-b[1]}) rotate(${(-yaw * 180) / Math.PI})` });
    el('ellipse', { cx: 0, cy: 0, rx: 0.13, ry: 0.24 }, robot);
    el('circle', { class: 'pm-head', cx: 0.02, cy: 0, r: 0.075 }, robot);
    el('path', { class: 'pm-face', d: 'M 0.135 -0.045 L 0.2 0 L 0.135 0.045 Z' }, robot);
    const sh = st.shoulder_xyz;
    el('line', { class: 'pm-reach', x1: sh[0], y1: -sh[1], x2: h[0], y2: -h[1], 'stroke-width': s * 0.45, 'stroke-dasharray': `${s * 0.6} ${s * 0.8}` });
    el('circle', { class: 'pm-shoulder', cx: sh[0], cy: -sh[1], r: s * 1.1 });
    label(`${st.arm} arm`, [sh[0], sh[1]], [s * 2, s * 4], s);
    const handle = el('circle', { class: 'pm-handle', r: s * 1.4 });
    label('handle', [h[0], h[1]], [s * 2.2, -s * 2.4], s);

    // animate the part along the twin's joint, the handle riding on it
    const pose = (f) => {
      const q = task.joint.q_closed + f * (task.joint.q_open - task.joint.q_closed);
      part.setAttribute('points', pts(hull(closed.map((p) => moved(task, p, q)))));
      const hp = moved(task, h, q);
      handle.setAttribute('cx', hp[0]); handle.setAttribute('cy', -hp[1]);
    };
    cancelAnimationFrame(raf);
    const loop = (t) => {
      if (!paused) phase = ((t - t0) / 5200) % 1;
      const f = phase < 0.5 ? phase * 2 : 2 - phase * 2; // open, then close
      pose(0.5 - 0.5 * Math.cos(Math.PI * f));
      raf = requestAnimationFrame(loop);
    };
    if (reduce) { pose(1); } else { t0 = performance.now() - phase * 5200; raf = requestAnimationFrame(loop); }
  }

  function label(text, at, off, s, anchor) {
    const t = el('text', { class: 'pm-label', x: at[0] + off[0], y: -at[1] + off[1], 'font-size': s * 2.7, 'text-anchor': anchor || 'start' });
    t.textContent = text;
  }

  function show(key) {
    const [obj, idx] = key.split(':');
    const task = data[obj].tasks[+idx];
    current = key;
    tabs.querySelectorAll('[role="tab"]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.key === key)));
    draw(task);
    const j = task.joint, st = task.stance;
    const span = Math.abs(j.q_open - j.q_closed);
    const open = j.type === 'prismatic' ? `${Math.round(span * 1000)} mm straight out` : `${Math.round((span * 180) / Math.PI)}° about the hinge`;
    const rows = [
      ['Action', task.action === 'pull' ? 'Pull along the measured slide' : 'Swing along the measured hinge'],
      ['Opening', open],
      ['Arm', `${st.arm[0].toUpperCase()}${st.arm.slice(1)}, ${task.path.length} waypoints`],
      ['Handle', `${task.handle[2].toFixed(2)} m up${st.object_lift > 0.01 ? `, object on a ${st.object_lift.toFixed(2)} m stand` : ''}`],
    ];
    readout.replaceChildren(...rows.flatMap(([k, v]) => { const dt = document.createElement('dt'); dt.textContent = k; const dd = document.createElement('dd'); dd.textContent = v; return [dt, dd]; }));
    const excerpt = { schema: task.schema, robot: task.robot, part: task.part, action: task.action, handle: task.handle,
      approach_normal: task.approach_normal, stance: { arm: st.arm, base_xyz: st.base_xyz, base_yaw: st.base_yaw },
      path: [task.path[0], task.path[1], '…', task.path[task.path.length - 1]] };
    code.textContent = JSON.stringify(excerpt, null, 1)
      .replace(/\[\n\s+(-?[\d.]+),\n\s+(-?[\d.]+),\n\s+(-?[\d.]+)\n\s+\]/g, '[$1, $2, $3]')
      .replace(/\{\n\s+"q": ([^,]+),\n\s+"handle": (\[[^\]]+\])\n\s+\}/g, '{"q": $1, "handle": $2}');
  }

  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    if (!paused) t0 = performance.now() - phase * 5200;
    pauseBtn.textContent = paused ? 'Play' : 'Pause';
    pauseBtn.setAttribute('aria-pressed', String(paused));
  });
  if (reduce) { pauseBtn.hidden = true; }

  fetch('/plans/plans.json').then((r) => r.json()).then((d) => {
    data = d;
    const buttons = [];
    for (const obj of Object.keys(d)) d[obj].tasks.forEach((t, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.key = `${obj}:${i}`;
      b.textContent = `${NAMES[obj] || obj} · ${t.part.replace(/_/g, ' ')}`;
      b.addEventListener('click', () => show(b.dataset.key));
      buttons.push(b);
    });
    tabs.replaceChildren(...buttons);
    tabs.addEventListener('keydown', (e) => {
      const i = buttons.findIndex((b) => b.dataset.key === current);
      const n = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : null;
      if (n === null) return;
      const b = buttons[(n + buttons.length) % buttons.length]; b.focus(); show(b.dataset.key);
    });
    show(`${Object.keys(d)[0]}:0`);
  }).catch(() => { readout.textContent = 'The plan data could not be loaded.'; });
})();
