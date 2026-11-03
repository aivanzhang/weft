// Scene polling carries child snapshots; accept only the membership established by that child's own response.
import assert from 'node:assert/strict';

export function selectedResponse(pathname, method, payload, jobID, binding = null) {
  const target = `/api/studio/jobs/${jobID}`;
  if ((pathname === target && method === 'GET') || (pathname === `${target}/start` && method === 'POST')) {
    const value = payload?.job || payload;
    assert.equal(value?.id, jobID, 'Selected job response changed ID');
    const scene = value.scene || null;
    if (scene) {
      assert.match(scene.id, /^[a-f0-9]{24}$/);
      assert.equal(typeof scene.asset_id, 'string');assert(scene.asset_id.length <= 64);
      assert.equal(typeof scene.asset_name, 'string');assert(scene.asset_name.length <= 120);
      assert.equal(typeof scene.title, 'string');assert(scene.title.length <= 120);
    }
    if (binding) assert.deepEqual(scene, binding, 'Selected job scene membership changed');
    return { value, binding: scene };
  }
  if (!binding || method !== 'GET' || pathname !== `/api/studio/scenes/${binding.id}`) return null;
  assert.equal(payload?.id, binding.id, 'Scene response changed ID');
  assert(Array.isArray(payload.assets) && payload.assets.length === 2, 'Expected a bounded two-asset scene response');
  const matches = payload.assets.filter(asset => asset.job?.id === jobID);
  assert.equal(matches.length, 1, 'Scene must contain the selected child exactly once');
  const asset = matches[0], value = asset.job;
  assert.equal(asset.id, binding.asset_id, 'Selected scene asset ID changed');
  assert.equal(asset.name, binding.asset_name, 'Selected scene asset name changed');
  assert.deepEqual(value.scene, binding, 'Selected child belongs to a different scene');
  return { value, binding };
}

export function observedBackendPath(pathname) {
  return /^\/api\/studio\/(session|(?:jobs|scenes)(?:\/[a-f0-9]{24})?)$/.test(pathname);
}
