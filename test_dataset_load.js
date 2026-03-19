const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = __dirname;
const PORT = 8765;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

function createStaticServer(rootDir, port) {
  const server = http.createServer((req, res) => {
    const requestPath = new URL(req.url, `http://127.0.0.1:${port}`).pathname;
    const safePath = requestPath === '/' ? '/app.html' : requestPath;
    const filePath = path.join(rootDir, safePath);

    if (!filePath.startsWith(rootDir)) {
      res.writeHead(403);
      res.end('Forbidden');
      return;
    }

    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(err.code === 'ENOENT' ? 404 : 500);
        res.end(err.code === 'ENOENT' ? 'Not found' : 'Server error');
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, {
        'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      res.end(data);
    });
  });

  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () => resolve(server));
  });
}

const EXPECTED_QUESTIONS = {
  pathvqa: 'is embolus derived from a lower-extremity deep venous thrombus lodged in a pulmonary artery branch?',
  vqarad: 'are regions of the brain infarcted?',
  omnimedvqa: 'What imaging technique was employed to obtain this picture?',
  pmcvqa: 'What is the uptake pattern in the breast?',
};

async function expectDatasetLoads(page, datasetValue) {
  const expectedQuestion = EXPECTED_QUESTIONS[datasetValue];
  await page.selectOption('#dsSrc', datasetValue);
  await page.waitForFunction((expected) => {
    const question = document.querySelector('#cQ')?.textContent?.trim();
    const options = document.querySelectorAll('#cOpts .mcq-i');
    return question === expected && options.length >= 2;
  }, expectedQuestion);

  const question = await page.locator('#cQ').textContent();
  const optionCount = await page.locator('#cOpts .mcq-i').count();
  const chips = await page.locator('#caseChips .sw-i').count();

  if (!question || !question.trim()) {
    throw new Error(`${datasetValue}: question is empty`);
  }
  if (question.trim() !== expectedQuestion) {
    throw new Error(`${datasetValue}: unexpected question "${question.trim()}"`);
  }
  if (optionCount < 2) {
    throw new Error(`${datasetValue}: expected at least 2 options, got ${optionCount}`);
  }
  if (chips < 1) {
    throw new Error(`${datasetValue}: expected at least 1 case type chip`);
  }

  console.log(`[PASS] ${datasetValue}`);
  console.log(`  Question: ${question.trim().slice(0, 80)}...`);
  console.log(`  Options: ${optionCount}`);
  console.log(`  Case types: ${chips}`);

  await page.click('button:has-text("Next")');
  await page.waitForFunction((previous) => {
    const question = document.querySelector('#cQ')?.textContent?.trim();
    return Boolean(question) && question !== previous;
  }, question.trim());

  const nextQuestion = await page.locator('#cQ').textContent();
  const pager = await page.locator('#casePager').textContent();
  if (!nextQuestion || nextQuestion.trim() === question.trim()) {
    throw new Error(`${datasetValue}: next-case navigation did not change the question`);
  }
  console.log(`  Next question: ${nextQuestion.trim().slice(0, 80)}...`);
  console.log(`  Pager: ${pager?.trim()}`);
}

async function expectMonteCarloTabVisible(page) {
  await page.click('button:has-text("3. Monte Carlo")');
  await page.waitForSelector('#mcStrategy');
  await page.waitForSelector('#mcMode');
  await page.waitForSelector('button:has-text("Run")');

  const heading = await page.locator('#pg2 .tile-h').textContent();
  if (!heading || !heading.includes('Monte Carlo')) {
    throw new Error(`Monte Carlo tab did not render correctly: "${heading || ''}"`);
  }

  console.log('[PASS] monte-carlo-tab');
}

async function main() {
  const server = await createStaticServer(ROOT, PORT);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const pageErrors = [];

  page.on('pageerror', (error) => {
    pageErrors.push(error.message);
  });

  try {
    console.log(`Opening http://127.0.0.1:${PORT}/app.html?fresh=test`);
    await page.goto(`http://127.0.0.1:${PORT}/app.html?fresh=test`, { waitUntil: 'networkidle' });

    await page.waitForSelector('#dsSrc');
    await expectMonteCarloTabVisible(page);
    await page.click('button:has-text("1. Problem")');
    await expectDatasetLoads(page, 'pathvqa');
    await expectDatasetLoads(page, 'vqarad');
    await expectDatasetLoads(page, 'omnimedvqa');
    await expectDatasetLoads(page, 'pmcvqa');

    if (pageErrors.length) {
      throw new Error(`Page errors detected: ${pageErrors.join(' | ')}`);
    }

    console.log('\nAll browser smoke checks passed.');
  } finally {
    await browser.close();
    await new Promise((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  }
}

main().catch((error) => {
  console.error('\nBrowser smoke test failed.');
  console.error(error.stack || error.message);
  process.exit(1);
});
