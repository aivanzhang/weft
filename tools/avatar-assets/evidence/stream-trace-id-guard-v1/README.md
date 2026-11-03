# Final stream trace ID guard

A malformed provider response ID containing the in-memory API credential is now
rejected before any trace update. No credential is stored in trace data.

Sixteen mocked transport tests pass, including the new prefixed/embedded-ID
regression. Both complete and interrupted HTTP/parser/runner integration checks
pass after the fix. These are local fixtures, not live provider or avatar results.

The earlier streaming/status bundle remains immutable. This additive bundle
binds the final module and tests. No provider call, ledger adjustment, backend
configuration change, lease release or new avatar generation occurred.
