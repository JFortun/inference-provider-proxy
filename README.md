# inference_provider_proxy (`ipp`)

A small native Python service for a home LAN. It sits in front of a **single-key** inference provider subscription, owns the one API key, and makes sure only one chat completion is ever in flight against the provider at a time. Overlapping requests from several machines queue and are served in arrival order instead of colliding with the provider's concurrency limit.

It speaks the OpenAI dialect, so any OpenAI-compatible client works by changing only the base address, and **no credential is needed**, because the deployment is a trusted home network.

## What you get

| Route                       | Gated? | Purpose                                                |
| --------------------------- | ------ | ------------------------------------------------------ |
| `POST /v1/chat/completions` | yes    | Chat, streaming and non-streaming.                     |
| `GET /v1/models`            | no     | Provider catalogue, short-TTL cached.                  |
| `GET /v1/usage`             | no     | Entitlement, cached and background-refreshed.          |
| `GET /dashboard`            | no     | One JSON document describing the whole session.        |
| `GET /`                     | no     | The same document as an HTML page (HTML + CSS, no JS). |
| `GET /healthz`              | no     | Cheap liveness; performs no provider call.             |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.15.

```sh
uv sync                 # create .venv and install pinned dependencies
cp .env.example .env    # then put your real provider key in .env
```

`.env` is git-ignored; `.env.example` contains placeholders only.

Every setting uses the `IPP_` prefix. The ones you are most likely to touch:

| Variable                          | Default                   | Meaning                                                   |
| --------------------------------- | ------------------------- | --------------------------------------------------------- |
| `IPP_UPSTREAM_BASE_URL`           | `https://api.example.com` | Provider address. Set it in `.env`.                       |
| `IPP_UPSTREAM_API_KEY`            | (none)                    | Your paid key. Never sent to clients, never logged.       |
| `IPP_BIND_HOST` / `IPP_BIND_PORT` | `0.0.0.0` / `13306`       | Where the proxy listens.                                  |
| `IPP_PROVIDER_CONCURRENCY`        | `1`                       | In-flight chat requests allowed.                          |
| `IPP_MAX_QUEUE_DEPTH`             | `64`                      | Waiters before new arrivals are rejected.                 |
| `IPP_QUEUE_TIMEOUT`               | `900`                     | Longest a request may wait for a slot.                    |
| `IPP_DISCONNECT_RELEASE`          | `15`                      | Bound on releasing the key after a mid-stream disconnect. |
| `IPP_MAX_RETRIES`                 | `2`                       | Bounded retry budget for transient provider failures.     |
| `IPP_LOG_LEVEL`                   | `info`                    | Level of the proxy's own log sections.                    |
| `IPP_LOG_COLOR`                   | `auto`                    | ANSI colors: `auto`, `always` or `never`.                 |

`.env.example` lists the full set (upstream/stream timeouts, cache TTLs, access logging, metrics retention).

## Run

```sh
uv run ipp                       # binds IPP_BIND_HOST:IPP_BIND_PORT
uv run ipp --host 0.0.0.0 --port 13306 --log-level info
uv run python -m inference_provider_proxy
```

Then, from another machine on the LAN:

```sh
curl http://proxy-host:13306/healthz
curl http://proxy-host:13306/v1/models
curl http://proxy-host:13306/dashboard

curl http://proxy-host:13306/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hello"}]}'
```

Streaming works the same way with `"stream": true`.

## Point a client at it

Any client that speaks the OpenAI dialect needs exactly three things:

1. **Base URL**: `http://proxy-host:13306/v1` (note the `/v1` suffix).
2. **Credential**: a placeholder. The proxy ignores whatever the client sends and substitutes the real key server-side, so any non-empty value (or the literal string `not-a-key`) satisfies clients that refuse to start without one.
3. **Model list**: take names from `GET /v1/models`; they are whatever the provider offers.

No client-side concurrency setting matters: requests simply wait in the queue at the proxy, regardless of how many the client fires at once.

## Distribution

`uv build` produces a wheel and an sdist with the standard PEP 517 backend:

```sh
uv build            # -> dist/*.whl and dist/*.tar.gz
```

The wheel is `py3-none-any`: it carries only the pure-Python package, and the runtime dependencies, including their platform-specific wheels, are resolved at install time, so the only prerequisite on the target is Python 3.15.

Install it as an isolated tool, which puts `ipp` on `PATH`:

```sh
uv tool install ./dist/*.whl
ipp --host 0.0.0.0 --port 13306
```

`uv tool install .` does the same straight from a checkout, and `dist/` is git-ignored. Long-running supervision (auto-start, restart after failure) is left to whatever the host already runs; this repo deliberately ships no service unit.

Every published release carries the same wheel and sdist as assets, built and attached by the [release workflow](.github/workflows/release.yml), so any tagged version installs straight from GitHub (swap `0.1.0` for the version you want):

```sh
uv tool install https://github.com/JFortun/inference-provider-proxy/releases/download/v0.1.0/inference_provider_proxy-0.1.0-py3-none-any.whl
```

To run a released version without installing it, `uvx` executes the wheel straight from the download cache; on a host with only [uv](https://docs.astral.sh/uv/) it also fetches the Python interpreter and the dependencies:

```sh
uvx --from https://github.com/JFortun/inference-provider-proxy/releases/download/v0.1.0/inference_provider_proxy-0.1.0-py3-none-any.whl ipp --host 0.0.0.0 --port 13306
```

## The dashboard

`GET /dashboard` returns one document with session identity, subscription status, gate state (including who currently holds the key and the ordered waiters), throughput, latency quantiles, an error breakdown by category, upstream status, per-client activity, and recent requests. Its field names are pinned by `src/inference_provider_proxy/contracts/dashboard.schema.json`.

The field to watch is `errors.by_kind.provider_busy`. It counts provider 429s after the retry budget; non-zero does **not** mean the gate failed: every `errors.recent[]` entry carries the `gate_in_flight` / `gate_capacity` recorded with it, and `in_flight <= capacity` there means another consumer is sharing the key.

## Two-machine check

On the proxy host, start the proxy. From a second machine on the same LAN, run five overlapping requests and confirm all five succeed:

```sh
IPP_SMOKE_BASE_URL=http://proxy-host:13306 uv run python qa/smoke/smoke.py
```

The script asserts every request returned `200`, that queue waits grew under contention, that `provider_busy` stayed `0`, and that the dashboard validates against the schema.

## Tests

```sh
uv run pytest                                        # unit, integration, architecture: 100% coverage gate
uv run pytest --run-e2e qa/e2e --no-cov              # real OpenAI-dialect client, real provider
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```

The `--run-e2e` suite talks to the real provider through a running proxy and consumes the single concurrency slot; it is skipped unless explicitly requested. Provider-side conditions are reported as skips, not failures.

Coverage is enforced at 100% over `src`, driven by the offline unit + integration suites (see `[tool.coverage]` in `pyproject.toml`). The e2e suite drives a separate running proxy over HTTP, so it is run with `--no-cov`.
