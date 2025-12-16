# agent-factory

Local-first multi-agent coding loop.

## What this does (v0)
- Calls LM Studio local API (OpenAI-compatible) for 4 roles: manager/dev/tester/researcher.
- Runs repo commands (install/lint/typecheck/tests) via strict allowlist.
- Produces artifacts + a unified diff patch file for inspection.

## Safety rules (DO NOT BREAK)
- Only run allowlisted commands from targets/*.yaml
- Never read/print secrets (.env, .ssh, keys)
- Never run destructive commands (rm -rf, del /q, format, disk ops)
- Keep diffs small (max files/lines set in target config)

## Quick start
1) Start LM Studio server on http://localhost:1234/v1 and load models.
2) Update models.yaml with real model IDs (curl /v1/models).
3) Update targets/mindbridge.yaml repo_path + commands to match your repo.
4) Run: python orchestrator/run_cycle.py

## Next upgrades (v1+)
- auto branch creation
- auto apply patch
- re-run tests
- auto commit + gh PR create
