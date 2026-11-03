// The unfunded bootstrap may prepare newly saved children, but never Start or retry.
export function admitPairMutation(method, pathname, children, prepareUnfunded, seen) {
  const create = pathname === '/api/studio/scenes';
  const plan = prepareUnfunded === true && children.some(id => /^[a-f0-9]{24}$/.test(id) && pathname === `/api/studio/jobs/${id}/plan`);
  if (method !== 'POST' || (!create && !plan) || seen.has(pathname)) throw new Error('Unexpected or duplicate pair mutation');
  seen.add(pathname);
  return create ? 'create' : 'plan';
}
