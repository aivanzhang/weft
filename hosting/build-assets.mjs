import { publicAssets } from "../website/public-assets.mjs";
import { copyFile, mkdir, readdir, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const output = resolve(here, ".build/public");
const assets = publicAssets;
async function existing(dir, prefix = "") {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const result = [];
  for (const entry of entries) {
    const path = prefix + entry.name;
    if (entry.isDirectory())
      result.push(...(await existing(resolve(dir, entry.name), path + "/")));
    else result.push(path);
  }
  return result;
}
const extras = (await existing(output)).filter(
  (name) => !assets.includes(name),
);
if (extras.length)
  throw new Error(
    "Unexpected public build files; inspect before deploying: " +
      extras.join(", "),
  );
for (const name of assets) {
  const target = resolve(output, name);
  await mkdir(dirname(target), { recursive: true });
  await copyFile(resolve(root, name), target);
}
const bytes = (
  await Promise.all(assets.map((name) => stat(resolve(output, name))))
).reduce((n, s) => n + s.size, 0);
console.log(
  `Bundled ${assets.length} allowlisted public files (${bytes} bytes). No uploads, credentials, or research runs included.`,
);
