import { publicAssets, publicPath } from "../website/public-assets.mjs";
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
// build_avatars.py owns website/avatars/ entirely and names files by content digest, so
// the guard below checks the prefix instead of pinning digests in the catalog.
const generated = "website/avatars/";
const generatedPublic = publicPath(generated);
const extras = (await existing(output)).filter(
  (name) =>
    !assets.map(publicPath).includes(name) && !name.startsWith(generatedPublic),
);
if (extras.length)
  throw new Error(
    "Unexpected public build files; inspect before deploying: " +
      extras.join(", "),
  );
const avatarFiles = (await existing(resolve(root, generated))).map(
  (name) => generated + name,
);
if (!avatarFiles.includes(generated + "index.json"))
  throw new Error(
    "Run build:avatars first; the hosted avatar catalog is missing from website/avatars/",
  );
const all = [...assets, ...avatarFiles];
for (const name of all) {
  const target = resolve(output, publicPath(name));
  await mkdir(dirname(target), { recursive: true });
  await copyFile(resolve(root, name), target);
}
const bytes = (
  await Promise.all(all.map((name) => stat(resolve(output, publicPath(name)))))
).reduce((n, s) => n + s.size, 0);
console.log(
  `Bundled ${assets.length} allowlisted public files plus ${avatarFiles.length} generated avatar files (${bytes} bytes). No uploads, credentials, or research runs included.`,
);
