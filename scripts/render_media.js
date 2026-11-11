// Render the site's product imagery from the real viewer (no stock, no mock-ups).
//   python scripts/dev_server.py 8765 &   then   node scripts/render_media.js
// Writes PNGs and a replay clip to web/img/ (scripts/build_site.py converts them).
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const BASE = process.env.OM_BASE || 'http://127.0.0.1:8765';
const OUT = path.join(__dirname, '..', 'web', 'img');

async function open(browser, name, w, h, opts = {}) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, colorScheme: 'light', ...opts });
  const page = await ctx.newPage();
  await page.goto(`${BASE}/gallery?embed=1#${name}`);
  await page.waitForFunction(() => typeof stage !== 'undefined' && stage.children.length > 0);
  await page.waitForTimeout(400);
  await page.evaluate(() => { playing = false; });
  return { ctx, page };
}

// what: 'points' | 'fitted' | 'proven' | 'full'
async function still(browser, name, file, { w, h, frame = 0, what = 'full', transparent = false, orbit = 1 }) {
  const { ctx, page } = await open(browser, name, w, h);
  await page.evaluate(({ frame, what, transparent, orbit }) => {
    showFrame(frame);
    stage.traverse((o) => {
      if (o.isMesh || o.isLineSegments) o.visible = what !== 'points';
      if (o.isMesh && what === 'proven') o.material.opacity = 0.45;
    });
    pointClouds.forEach((p) => { p.visible = what === 'points' || what === 'proven' || what === 'full'; p.material.size = what === 'points' ? 0.016 : what === 'proven' ? 0.013 : 0.006; if (what === 'points') p.material.color.set(0x2f3a0c); });
    axisHelpers.forEach((a) => { a.visible = what === 'fitted' || what === 'full'; });
    scene.children.forEach((c) => { if (c.type === 'GridHelper') c.visible = !transparent; });
    if (transparent) { document.body.style.background = 'transparent'; renderer.setClearColor(0x000000, 0); }
    camera.position.sub(controls.target).multiplyScalar(orbit).add(controls.target);
    controls.update(); renderer.render(scene, camera);
  }, { frame, what, transparent, orbit });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT, file), omitBackground: transparent });
  await ctx.close();
  console.log('still', file);
}

async function clip(browser, name, file, { w, h, seconds }) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, recordVideo: { dir: OUT, size: { width: w, height: h } } });
  const page = await ctx.newPage();
  await page.goto(`${BASE}/gallery?embed=1#${name}`);
  await page.waitForFunction(() => typeof stage !== 'undefined' && stage.children.length > 0);
  await page.evaluate(() => {
    playing = true; f = 0;
  });
  await page.waitForTimeout(seconds * 1000);
  const video = page.video();
  await ctx.close();
  fs.renameSync(await video.path(), path.join(OUT, file));
  console.log('clip', file);
}

(async () => {
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  await still(browser, 'kitchen_cabinet', 'stage-tracked.png', { w: 700, h: 700, frame: 11, what: 'points', transparent: true, orbit: 0.62 });
  await still(browser, 'kitchen_cabinet', 'stage-fitted.png', { w: 700, h: 700, frame: 11, what: 'fitted', transparent: true, orbit: 0.62 });
  await still(browser, 'kitchen_cabinet', 'stage-proven.png', { w: 700, h: 700, frame: 11, what: 'proven', transparent: true, orbit: 0.62 });
  await still(browser, 'dresser', 'square-dresser.png', { w: 800, h: 800, frame: 22, what: 'full', orbit: 0.8 });
  await still(browser, 'desk_lamp', 'square-desk-lamp.png', { w: 800, h: 800, frame: 0, what: 'full', orbit: 1.25 });
  await still(browser, 'fridge', 'square-fridge.png', { w: 800, h: 800, frame: 20, what: 'full', orbit: 0.62 });
  await still(browser, 'kitchen_cabinet', 'hero-poster.png', { w: 760, h: 620, frame: 16, what: 'full' });
  await clip(browser, 'dresser', 'replay-dresser.webm', { w: 720, h: 720, seconds: 5.2 });
  await browser.close();
})();
