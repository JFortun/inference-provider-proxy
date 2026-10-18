# Architecture

Hexagonal architecture (ports and adapters): the core never touches HTTP or the
provider directly. Everything crosses a port.

```mermaid
flowchart LR
    subgraph driving["Driving side"]
        C["Clients<br/>OpenAI dialect · dashboard"]
        IN["Inbound adapters<br/>FastAPI routes · middleware · error handlers"]
    end

    subgraph core["Application core"]
        direction LR
        P1["Inbound ports<br/>ChatCompletionUseCase<br/>CatalogUseCase · DashboardUseCase"]
        SVC["Services<br/>ChatService · CatalogService<br/>DashboardService"]
        P2["Driven ports<br/>LLMProviderPort · ConcurrencyGatePort<br/>MetricsPort · SubscriptionPort · ClockPort"]
        DOM[("Domain<br/>models · errors · metrics<br/>pure: no I/O, no framework")]
    end

    subgraph driven["Driven side"]
        OUT["Outbound adapters<br/>AsyncioGate · Provider adapter (httpx)<br/>MemoryMetrics · SubscriptionPoller"]
        W["Upstream provider"]
    end

    C --> IN --> P1 --> SVC --> P2 --> OUT --> W
    DOM -. everything in the core<br/>speaks domain types .-> SVC
```

Ports are `typing.Protocol` contracts in `application/ports/`; the services and
adapters are constructed with them as keyword arguments. `main.py` is the single
[composition root](../src/inference_provider_proxy/main.py): the only place that
knows both the ports and the concrete adapters.

## Traffic flow

```mermaid
sequenceDiagram
    participant C as Client
    participant MW as Inbound adapter<br/>(middleware + route)
    participant S as ChatService
    participant G as ConcurrencyGatePort
    participant P as LLMProviderPort
    participant U as Upstream provider

    C->>MW: POST /v1/chat/completions
    MW->>S: complete(ChatRequest) via ChatCompletionUseCase
    S->>G: slot(...)
    Note over G: FIFO wait, queue_full / queue_timeout<br/>raise ProxyError before any provider call
    G-->>S: QueueTicket (admitted)
    S->>P: complete(body)
    P->>U: HTTPS with the real key (substituted server-side)
    U-->>P: response, or a failure classified into one ErrorKind
    P-->>S: ProviderCompletion / ProxyError
    S-->>MW: ChatResult (queue_wait_ms, payload)
    MW-->>C: 200 JSON, or an OpenAI-shaped error envelope
    Note over S: the gate slot is released only after<br/>the last byte is delivered
```

Streaming follows the same path through `stream()`; the slot stays held until
the chunk iterator finishes or is closed.

## Rules

| Layer                | May depend on           | Must never depend on                     |
| -------------------- | ----------------------- | ---------------------------------------- |
| `domain/`            | standard library only   | anything above it, any framework         |
| `application/`       | `domain`                | FastAPI, httpx, adapters                 |
| `adapters/inbound/`  | `application`, `domain` | `adapters/outbound/`                     |
| `adapters/outbound/` | `application`, `domain` | `adapters/inbound/`                      |
| `main.py`            | everything              | nothing (it is the one place allowed to) |

1. **Layering**: dependencies point inward only: `adapters → application → domain`.
2. **Domain purity**: no `fastapi`, `starlette`, `httpx`, `pydantic` or `uvicorn` inside `domain/`.
3. **Inbound never reaches outbound**: routes see ports, never concrete adapters.
4. **No import cycles** at module level.

All four are enforced by [`qa/architecture/test_hexagon.py`](../qa/architecture/test_hexagon.py),
each with a companion test proving it fails when deliberately violated.
