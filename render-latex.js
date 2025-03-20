const katex = require('katex');
const puppeteer = require('puppeteer');
const execSync = require('child_process').execSync;
const fs = require('fs');

(async () => {
  const latexInput = process.argv[2];

  const html = `<!DOCTYPE html>
  <html>
  <head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css">
  <style>
    body { display: flex; justify-content: center; align-items: center; margin:0; background: transparent; }
    .katex { font-size: 3em; padding: 10px; }
  </style>
  </head>
  <body>${katex.renderToString(latexInput, { throwOnError: false, displayMode: true })}</body>
  </html>`;

  fs.writeFileSync('temp.html', html);

  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();
  await page.setContent(html);

  const katexElement = await page.$('.katex');
  const clip = await katexElement.boundingBox();

  // Manually expand clip dimensions by 10 pixels on all sides
  const padding = 30;
  
  const adjustedClip = {
    x: clip.x - padding,
    y: clip.y - padding,
    width: clip.width + padding * 2,
    height: clip.height + padding * 2,
  };
  
  await page.screenshot({
    path: 'equation.png',
    clip: adjustedClip,
    omitBackground: true,
  });
  
  await browser.close();

  execSync('powershell Set-Clipboard -Path "equation.png"');

  console.log('✅ Equation rendered & copied to clipboard!');
})();
