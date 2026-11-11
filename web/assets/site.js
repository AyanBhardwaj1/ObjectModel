// Shared behaviour: theme toggle (remembered per browser), copy buttons, entrance reveals.
(function () {
  const root = document.documentElement;
  root.classList.add('js');
  try { const t = localStorage.getItem('om-theme'); if (t) root.dataset.theme = t; } catch (e) { /* storage blocked: follow the system */ }

  document.addEventListener('click', (ev) => {
    const tog = ev.target.closest('[data-theme-toggle]');
    if (tog) {
      const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      tog.setAttribute('aria-pressed', String(!dark));
      try { localStorage.setItem('om-theme', root.dataset.theme); } catch (e) { /* not persisted */ }
    }
    const btn = ev.target.closest('[data-copy]');
    if (btn && navigator.clipboard) {
      navigator.clipboard.writeText(btn.getAttribute('data-copy')).then(() => {
        const old = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => (btn.textContent = old), 1400);
      }, () => { /* clipboard refused: the text stays selectable */ });
    }
  });

  // Reveal once on entry; anything already on screen shows immediately.
  const items = document.querySelectorAll('.rise, .shutter');
  if (!('IntersectionObserver' in window) || matchMedia('(prefers-reduced-motion: reduce)').matches) {
    items.forEach((el) => el.classList.add('in'));
    return;
  }
  // A shutter starts fully clipped, which observers can read as zero area, so watch its
  // (unclipped) parent instead and open the shutter when that arrives.
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      const target = e.target.__omShutter || e.target;
      target.classList.add('in');
      io.unobserve(e.target);
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0 });
  items.forEach((el) => {
    if (el.classList.contains('shutter') && el.parentElement) {
      el.parentElement.__omShutter = el;
      io.observe(el.parentElement);
    } else {
      io.observe(el);
    }
  });
  // Safety net: never leave content hidden if an observer callback is missed.
  setTimeout(() => items.forEach((el) => el.classList.add('in')), 4000);
})();
