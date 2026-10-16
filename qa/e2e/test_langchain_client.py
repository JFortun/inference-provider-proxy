"""End-to-end checks driven by a *real* LangChain client (R10 / T052).

These talk to a running proxy and therefore to the real provider, so they are
gated behind ``--run-e2e`` (see ``qa/conftest.py``) and are serial and slow by
design: they consume the single concurrency slot.

Provider-side conditions (out of reserved window, edge block, entitlement) are
reported as *skips*; only proxy-side assertions fail the suite.

Each test closes its own chat client before finishing. LangChain's ``ChatOpenAI``
owns an ``httpx`` connection pool bound to the running event loop, and if it is
only garbage-collected later — on a subsequent test's loop — teardown raises
``RuntimeError: Event loop is closed``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator

import pytest

pytestmark = [
    pytest.mark.e2e,
    # One event loop for the whole module: LangChain's chat client owns an httpx
    # connection pool bound to the loop it was created on, and a per-test loop
    # makes that pool close onto a dead loop during the next test's teardown.
    pytest.mark.asyncio(loop_scope="session"),
]

BASE_URL = os.environ.get("IPP_E2E_BASE_URL", "http://127.0.0.1:13306/v1")
MODEL = os.environ.get("IPP_E2E_MODEL", "deepseek-v4.1-flash")

_PROVIDER_SIDE_MARKERS = (
    "402",
    "outside its reserved",
    "payment",
    "edge",
    "1010",
    "entitlement",
    "invalid_api_key",
)


def _skip_if_provider_side(exc: Exception) -> None:
    text = str(exc).lower()
    if any(marker.lower() in text for marker in _PROVIDER_SIDE_MARKERS):
        pytest.skip(f"provider-side condition, not a proxy fault: {exc}")


async def _aclose(llm: object) -> None:
    """Close the underlying async client while this test's loop is still alive."""
    with contextlib.suppress(Exception):
        client = getattr(llm, "async_client", None)
        close = getattr(client, "close", None)
        if close is not None:
            await close()


@contextlib.asynccontextmanager
async def _llm(**kwargs: object) -> AsyncIterator[object]:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key="not-needed", **kwargs)
    try:
        yield llm
    finally:
        await _aclose(llm)


async def _drive(call):
    try:
        return await call()
    except Exception as exc:  # noqa: BLE001 - classify provider-side vs proxy-side
        _skip_if_provider_side(exc)
        raise


async def test_non_streaming_invoke() -> None:
    async with _llm() as llm:
        result = await _drive(lambda: llm.ainvoke("Reply with the single word: pong"))
    assert result.content


async def test_streaming_invoke() -> None:
    chunks: list[str] = []
    try:
        async with _llm() as llm:
            async for chunk in llm.astream("Count from one to three."):
                if chunk.content:
                    chunks.append(str(chunk.content))
    except Exception as exc:  # noqa: BLE001
        _skip_if_provider_side(exc)
        raise
    assert chunks, "streaming produced no incremental content"


async def test_tool_calling() -> None:
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    async with _llm() as llm:
        bound = llm.bind_tools([get_weather])
        result = await _drive(lambda: bound.ainvoke("What is the weather in Paris?"))
    assert getattr(result, "tool_calls", None) is not None


async def test_langgraph_agent_loop() -> None:
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:  # noqa: BLE001
        pytest.skip("langgraph is not importable in this environment")

    from typing import TypedDict

    class State(TypedDict):
        question: str
        answer: str

    async def answer(state: State) -> State:
        async with _llm() as llm:
            result = await _drive(lambda: llm.ainvoke(state["question"]))
        return {"question": state["question"], "answer": str(result.content)}

    graph = StateGraph(State)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    compiled = graph.compile()

    final = await compiled.ainvoke({"question": "Say 'agent ok'.", "answer": ""})
    assert final["answer"]


async def test_five_concurrent_chains_all_succeed() -> None:
    # Distinct clients, so the concurrency is real (five separate HTTP callers).
    async with contextlib.AsyncExitStack() as stack:
        llms = [await stack.enter_async_context(_llm()) for _ in range(5)]

        async def one(llm: object, index: int) -> str:
            result = await _drive(lambda: llm.ainvoke(f"Reply with the number {index}."))
            return str(result.content)

        answers = await asyncio.gather(*(one(llm, i) for i, llm in enumerate(llms)))
    assert len(answers) == 5
    assert all(answers)


async def test_midstream_abort_then_new_request() -> None:
    async with _llm() as llm:
        stream = llm.astream("Write a long paragraph about rivers.")
        try:
            first = await anext(stream.__aiter__())
            assert first is not None
        except Exception as exc:  # noqa: BLE001
            _skip_if_provider_side(exc)
            raise
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

        # The proxy must have released the slot, so this completes promptly.
        follow_up = await _drive(lambda: llm.ainvoke("Reply with: recovered"))
    assert follow_up.content


async def test_long_context_round_trip() -> None:
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    async with _llm() as llm:
        result = await _drive(lambda: llm.ainvoke(f"{filler}\n\nReply with: done"))
    assert result.content
