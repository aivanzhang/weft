// A completed mutation precedes status observation; the old DOM is not a job result.
export async function observePreparation({ receivePost, click, readJob, savePost, pause,
  now = () => performance.now(), maxMs = 90000 }) {
  const post = receivePost();
  const [response] = await Promise.all([post, Promise.resolve().then(click)]);
  await savePost(response);
  if (!response.ok) throw new Error(`Prepare request failed (${response.status})`);
  const deadline = now() + maxMs;
  while (now() < deadline) {
    const job = await readJob();
    if (["planned", "needs_inspection", "failed", "unsupported", "awaiting_authorization", "cancelled"].includes(job.status)) return job;
    await pause(250);
  }
  throw new Error("Preparation observation timed out; preserve the detached worker and inspect custody before retrying");
}
