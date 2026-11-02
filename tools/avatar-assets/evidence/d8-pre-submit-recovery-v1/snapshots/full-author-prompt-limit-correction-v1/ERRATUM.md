# Correction: complete author-prompt limit

The prior prompt-admission conclusion was wrong. `outputs/visual-repair-continuation-tests-v1/actual-result.json` checked 137,636 bytes including an allowance against the 160 KiB serialized-data limit. The real API runner imports `agent_runner.MAX_PROMPT_BYTES`, which limits the **complete prompt to 128 KiB (131,072 bytes)**. That earlier estimate exceeded the actual limit by **6,564 bytes**. Its original report, manifests and compact evidence bundle remain unchanged. The successful read-only custody checks remain valid; they did not prove prompt admission.

The actual saved d8 continuation reconstructs to **136,896 bytes before its directive and 137,155 bytes including it**, so it correctly fails the existing runner limit. No threshold was increased.

The revised serializer changes JSON whitespace only. An independent comparison confirms equal decoded data, unchanged prompt logic outside that serialization, and the exact continuation directive from both the saved and current `Pipeline._run` syntax trees. The new complete prompt is **84,075 / 131,072 bytes**. With a conservative 1,024-byte allowance instead of the exact 259-byte directive, it is **84,840 / 131,072 bytes**. Its JSON payload is separately **56,082 / 163,840 bytes**.

`result.json` records exact source/input/prior-evidence hashes, byte counts and canonical payload equality. Original saved inputs and prior evidence hashes are unchanged. This check made no provider, native, worker, release or budget calls and did not create a successor job. It is an input-admission proof, not a successful generation or appearance review.

The parent updated `tests/test_repair_prompt_view.py` to import the runner limit and reported six passing focused tests. That test source is snapshotted here; its tests were not redundantly rerun in this independent lane. `verify.py` and its actual bounded-job log provide the independent actual-data measurement.
