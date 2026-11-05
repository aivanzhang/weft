// Both servers expose only this catalog, never the repository or private runs.
export const publicAssets = [
  "website/index.html",
  "website/references.html",
  "website/references.css",
  "website/styles.css",
  "website/site.js",
  "website/contents.css",
  "website/contents.js",
  "website/footer.css",
  "website/footer.js",
  "website/environments.css",
  "website/environments.js",
  "website/environments/catalog.js",
  "website/environments/rolling-ball.json",
  "website/environments/rolling-ball.jpg",
  "website/fonts/InterVariable.woff2",
  "website/fonts/OFL.txt",
  "assets/brand/source/weft-wordmark.png",
  "assets/brand/exports/weft-mascot.png",
  "src/weft/assets/engine.js",
  // The demo page itself, served as it exists in the repository rather than reimplemented.
  "src/weft/assets/demo_build.html",
  "src/weft/assets/demo.css",
  "src/weft/assets/brand.css",
  "src/weft/assets/demo_build.js",
  "src/weft/assets/demo_focus.js",
  "src/weft/assets/demo_studio.js",
  "src/weft/assets/demo.js",
  "src/weft/assets/weft-logo.png",
  "data/rolling-ball.mp4",
  // Curated showcase media the demo page plays; licensed and credited per seed.
  "assets/demo/motion-showcase/bodyweight-squats/result.mp4",
  "assets/demo/motion-showcase/bodyweight-squats/source.mp4",
  "assets/demo/motion-showcase/deadlift/result.mp4",
  "assets/demo/motion-showcase/deadlift/source.mp4",
  "assets/demo/motion-showcase/incline-press/result.mp4",
  "assets/demo/motion-showcase/incline-press/source.mp4",
  "assets/demo/motion-showcase/jumping-jacks/result.mp4",
  "assets/demo/motion-showcase/jumping-jacks/source.mp4",
  "assets/demo/motion-showcase/lunge-twist/result.mp4",
  "assets/demo/motion-showcase/lunge-twist/source.mp4",
  "assets/demo/motion-showcase/overhead-squat/result.mp4",
  "assets/demo/motion-showcase/overhead-squat/source.mp4",
  "assets/demo/motion-showcase/single-leg-squat/result.mp4",
  "assets/demo/motion-showcase/single-leg-squat/source.mp4",
  "assets/demo/motion-showcase/sots-press/result.mp4",
  "assets/demo/motion-showcase/sots-press/source.mp4",
  "assets/demo/motion-showcase/waiters-squat/result.mp4",
  "assets/demo/motion-showcase/waiters-squat/source.mp4",
  "assets/demo/previews/rolling-result.mp4",
  "assets/demo/previews/rolling-source.mp4",
];

// The site is served from the root, so website/ files lose that prefix on the way out while
// assets/, src/ and data/ keep their repository paths. Both the deployed bundle and the dev
// server map through here, so the two cannot drift.
const flattened = ["website/", "src/weft/assets/"];
export const publicPath = (name) => {
  const prefix = flattened.find((value) => name.startsWith(value));
  return prefix ? name.slice(prefix.length) : name;
};

// Public URL path -> repository path, for a server resolving a request back to a file.
export const publicRoutes = new Map(
  publicAssets.map((name) => [`/${publicPath(name)}`, name]),
);
