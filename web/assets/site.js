// Theme toggle (remembered per browser) and copy buttons.
(function () {
  const root = document.documentElement;
  try { const t = localStorage.getItem('om-theme'); if (t) root.dataset.theme = t; } catch (e) {}
  document.addEventListener('click', (ev) => {
    const tog = ev.target.closest('[data-theme-toggle]');
    if (tog) {
      const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      try { localStorage.setItem('om-theme', root.dataset.theme); } catch (e) {}
    }
    const btn = ev.target.closest('[data-copy]');
    if (btn) {
      const text = btn.getAttribute('data-copy');
      const done = () => { const old = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => (btn.textContent = old), 1400); };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
    }
  });
})();
