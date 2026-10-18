# AGENTS.md

Guidance for coding agents working on `inference-provider-proxy` (`ipp`).

This file is deliberately short: it holds only what has no other home. When a topic already has a source of truth, follow the link instead of copying the content here.

## Sources of truth

| Topic                                                                     | Source                                                                                                                           |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| What the service does, routes, configuration, running it, client setup    | [README.md](README.md)                                                                                                           |
| Every `IPP_` environment variable, defaults and comments                  | [.env.example](.env.example)                                                                                                     |
| Architecture, the four layering rules, traffic flow diagrams              | [docs/architecture.md](docs/architecture.md)                                                                                     |
| Dependencies (exact pins), entry point, pytest / coverage / ruff settings | [pyproject.toml](pyproject.toml)                                                                                                 |
| Test suites, lint, type-check and build commands                          | [README.md#tests](README.md#tests)                                                                                               |
| Hexagonal boundary enforcement, one proof per rule                        | [qa/architecture/test_hexagon.py](qa/architecture/test_hexagon.py)                                                               |
| Dashboard wire contract, field names                                      | [src/inference_provider_proxy/contracts/dashboard.schema.json](src/inference_provider_proxy/contracts/dashboard.schema.json)     |
| Release process, tag/version check                                        | [.github/workflows/release.yml](.github/workflows/release.yml)                                                                   |
| Commit message format                                                     | [.agents/skills/conventional-commit/SKILL.md](.agents/skills/conventional-commit/SKILL.md), with scope vocabulary from `git log` |

## Before you finish

No workflow in [.github/workflows](.github/workflows) runs tests, lint or type checks; the only one builds release artifacts. Treat the commands in [README.md#tests](README.md#tests) as your own pre-flight and run them locally before reporting done. The suite enforces 100% coverage over `src`, so changed or new source code needs tests or the run fails.

## Rules with no other home

- **Do not weaken or delete a test to make a change pass.** Fix the code, or stop and explain the conflict first.
- **Never stage `.env`** or a real credential, and never hardcode the provider key in source, tests or fixtures; `.env.example` stays placeholders only (see [README.md#setup](README.md#setup)).
- **Client authentication is a design decision, not a local improvement.** The proxy is deliberately open on a trusted LAN; ask before introducing credentials.
- **Never push, amend or rebase an existing commit** unless explicitly asked (the [commit skill](.agents/skills/conventional-commit/SKILL.md) says the same).
