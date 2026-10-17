# Provider probe fixtures

Bodies captured from the live provider as described in `PLAN.md` §3. They are
frozen here so the error-mapping tests (T045) do not depend on the provider
being reachable. `code` values are strings, exactly as the provider returns them
(`PLAN.md` §3.1): matching must stay string-safe.

| File | Status | Shape | Expected category |
|------|--------|-------|-------------------|
| `throttling_429.json` | 429 | JSON `error.type = throttling_error`, `code = "429"`, `retry-after` header | `provider_busy` |
| `auth_401.json` | 401 | JSON error | `upstream_auth` |
| `edge_block_403.txt` | 403 | non-JSON, 17-byte `error code: 1010` | `edge_block` |
| `not_reserved_402.json` | 402 | JSON error | `not_reserved` |
| `upstream_500.json` | 500 | JSON error | `upstream_5xx` |
| `unknown_418.json` | 418 | JSON error of an unrecognised shape | `unknown` |
