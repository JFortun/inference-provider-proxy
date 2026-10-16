"""Live acceptance smoke run (T051).

Fires N concurrent chat requests at a *running* proxy and asserts the gate did
its job. This is a raw-``httpx`` script, deliberately independent of the pytest
integration suite, so it can be pointed at a real deployment.

Usage::

    IPP_SMOKE_BASE_URL=http://proxy-host:13306 \\
    IPP_SMOKE_MODEL=deepseek-v4.1-flash \\
    uv run python qa/smoke/smoke.py

Exit code is non-zero if any assertion fails.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from importlib.resources import files

import httpx

BASE_URL = os.environ.get("IPP_SMOKE_BASE_URL", "http://127.0.0.1:13306").rstrip("/")
MODEL = os.environ.get("IPP_SMOKE_MODEL", "deepseek-v4.1-flash")
CONCURRENCY = int(os.environ.get("IPP_SMOKE_CONCURRENCY", "5"))
SCHEMA = json.loads(
    files("inference_provider_proxy.contracts").joinpath("dashboard.schema.json").read_text()
)


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


async def _chat(client: httpx.AsyncClient, index: int, *, stream: bool) -> httpx.Response:
    if stream:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": f"say hello ({index})"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])
        assert response.status_code == 200, response.status_code
        if "data: [DONE]" not in body:
            _fail("streaming response did not terminate with [DONE]")
        return response
    return await client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": f"say hello ({index})"}]},
    )


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        health = await client.get("/healthz")
        if health.status_code != 200:
            _fail(f"/healthz returned {health.status_code}")
        print(f"ok: /healthz at {BASE_URL}")

        responses = await asyncio.gather(
            *(_chat(client, i, stream=False) for i in range(CONCURRENCY))
        )
        if any(r.status_code != 200 for r in responses):
            _fail(f"concurrent chat failed: {[r.status_code for r in responses]}")
        print(f"ok: {CONCURRENCY} concurrent requests all returned 200")

        waits = [int(r.headers.get("X-Queue-Wait-Ms", "0")) for r in responses]
        if max(waits) <= 0:
            _fail("X-Queue-Wait-Ms did not grow under contention")
        print(f"ok: queue wait grew under contention (max {max(waits)}ms)")

        streamed = await _chat(client, 99, stream=True)
        if streamed.status_code != 200:
            _fail(f"streaming request returned {streamed.status_code}")
        print("ok: streaming round-trip")

        models = await client.get("/v1/models")
        if models.status_code != 200:
            _fail(f"/v1/models returned {models.status_code}")
        print("ok: /v1/models")

        dashboard = await client.get("/dashboard")
        if dashboard.status_code != 200:
            _fail(f"/dashboard returned {dashboard.status_code}")
        document = dashboard.json()

        try:
            import jsonschema

            jsonschema.validate(document, SCHEMA)
            print("ok: /dashboard matches contracts/dashboard.schema.json")
        except ImportError:
            print("skip: jsonschema not installed, dashboard not validated against the schema")

        by_kind = document["errors"]["by_kind"]
        if by_kind["provider_busy"] != 0:
            _fail(f"provider_busy is {by_kind['provider_busy']}, expected 0 (SC-002)")
        if document["gate"]["in_flight"] > document["gate"]["capacity"]:
            _fail("gate.in_flight exceeded gate.capacity (SC-003)")
        print("ok: provider_busy stayed 0 and in-flight never exceeded capacity")

    print("SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())
