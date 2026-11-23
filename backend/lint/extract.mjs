// The SPA is one inline <script> in index.html, served as /app.js (see
// _spa_script in app/main.py). ESLint cannot read it inside HTML, so lint the
// same block main.py serves: the largest attribute-less <script>.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const html = readFileSync(new URL("../app/static/index.html", import.meta.url), "utf8");
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
const app = blocks.reduce((a, b) => (b.length > a.length ? b : a), "");
if (app.length < 100000) {
  console.error(`extract.mjs: the SPA script is ${app.length} chars; did index.html change shape?`);
  process.exit(1);
}
mkdirSync(new URL("build/", import.meta.url), { recursive: true });
writeFileSync(new URL("build/app.js", import.meta.url), app);
