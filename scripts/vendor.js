// Copies the browser builds of the UI libraries from node_modules into ui/vendor.
// The UI has no bundler: ui/ is served as-is by Tauri, so vendored files are committed.
// Run after upgrading a dependency:  npm run vendor
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const out = path.join(root, "ui", "vendor");
const files = [
  ["node_modules/marked/lib/marked.umd.js", "marked.umd.js"],
  ["node_modules/dompurify/dist/purify.min.js", "purify.min.js"],
  ["node_modules/@highlightjs/cdn-assets/highlight.min.js", "highlight.min.js"],
  ["node_modules/@highlightjs/cdn-assets/styles/github-dark.min.css", "hljs-dark.css"],
  // math in replies ($…$, $$…$$) and diagrams (```mermaid)
  ["node_modules/katex/dist/katex.min.js", "katex.min.js"],
  ["node_modules/katex/dist/katex.min.css", "katex.min.css"],
  ["node_modules/mermaid/dist/mermaid.min.js", "mermaid.min.js"],
];

fs.mkdirSync(out, { recursive: true });
for (const [from, to] of files) {
  fs.copyFileSync(path.join(root, from), path.join(out, to));
  console.log(`vendored ${to}`);
}
// KaTeX's fonts (its CSS loads fonts/*.woff2 first; every webview Lumen runs in reads woff2)
const fonts = path.join(out, "fonts");
fs.mkdirSync(fonts, { recursive: true });
for (const f of fs.readdirSync(path.join(root, "node_modules/katex/dist/fonts")).filter((f) => f.endsWith(".woff2"))) {
  fs.copyFileSync(path.join(root, "node_modules/katex/dist/fonts", f), path.join(fonts, f));
}
console.log("vendored KaTeX fonts");
