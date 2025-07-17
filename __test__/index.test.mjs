import { expect, test } from 'vitest'

import { templateMatch } from '../index.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const needleScale = 1;

function readImage(filePath) {
  return fs.readFileSync(path.join(__dirname, 'fixtures', filePath));
}

test('match template', () => {
  const haystackBuffer = readImage('haystack.png');
  const needleBuffer = readImage('needle.png');
  const result = templateMatch(haystackBuffer, needleBuffer, {
    scale: 1 / needleScale,
  });
  console.log(result);
  expect(result.found).toBe(true);
});