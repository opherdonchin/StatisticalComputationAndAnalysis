const katex = require('katex');
const puppeteer = require('puppeteer');
const execSync = require('child_process').execSync;
const fs = require('fs');

(async () => {
  const html = `<!DOCTYPE html>
  <html>
  <head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css">
  <style>
    body { display: flex; justify-content: center; align-items: center; height: 100vh; margin:0; background: transparent; }
    .katex { font-size: 3em; }
  </style>
  </head>
  <body>${katex.renderToString("'$latexInput'", { throwOnError: false, displayMode: true })}</body>
  </html>`;

  fs.writeFileSync('temp.html', html);

  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();
  await page.setContent(html);
  const elementHandle = await page.$('body');
  const clip = await elementHandle.boundingBox();

  await page.screenshot({
    path: 'equation.png',
    clip: clip,
    omitBackground: true,
  });

  await browser.close();

  execSync('powershell Set-Clipboard -Path "equation.png"');

  console.log('✅ Equation rendered & copied to clipboard!');
})();
