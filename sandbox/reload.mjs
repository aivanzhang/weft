import { watch } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

export function changeKind(relative) {
  const name = relative.split(path.sep).join('/');
  if (/(^|\/)(\.[^/]+|__pycache__|node_modules)(\/|$)/.test(name) || /\.(test|integration\.test)\.mjs$/.test(name)) return null;
  if (/^src\/weft\/.*\.py$/.test(name) || name === 'src/weft/assets/demo_examples.json') return 'backend';
  if (/^(website|src\/weft\/assets|assets\/demo|assets\/brand\/exports)\/.*\.(html|css|js|cjs|json|png|jpg|svg|woff2|mp4)$/.test(name)) return 'page';
  return null;
}

export function watchSources(root, changed, { watchFile = watch, delay = 180, log = console } = {}) {
  let timer, pending = null;
  const watchers = [];
  const close = () => { clearTimeout(timer); watchers.forEach(watcher => watcher.close()); };
  try {
    for (const folder of ['website', 'src/weft', 'assets/demo', 'assets/brand/exports']) {
      const watcher = watchFile(path.join(root, folder), { recursive: true }, (_, filename) => {
        if (!filename) return;
        const kind = changeKind(path.join(folder, String(filename)));
        if (!kind) return;
        pending = pending === 'backend' ? pending : kind;
        clearTimeout(timer);
        timer = setTimeout(() => { const next = pending; pending = null; changed(next); }, delay);
      });
      watcher.on('error', error => log.warn(`Hot refresh watcher: ${error.message}`));
      watchers.push(watcher);
    }
  } catch (error) { close(); throw error; }
  return close;
}

export function reloadState() {
  let revision = randomUUID(), ready = false;
  return {
    status: () => ({ revision, ready }),
    pending() { ready = false; },
    refresh() { revision = randomUUID(); },
    publish() { revision = randomUUID(); ready = true; },
    inject(html) {
      const tag = `<script src="/__weft_reload.js" data-version="${revision}" defer></script>`;
      return html.replace(/<\/body>/i, tag + '</body>');
    },
  };
}
