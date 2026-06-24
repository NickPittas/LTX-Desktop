# Handoff — LTX-Desktop local model profiles
**Date:** 2026-06-24

## Goal
Build an LTX-Desktop fork that does not force model downloads and can use user-selected local LTX-2.3 components: official monolithic checkpoints first, then official adapters, Kijai split safetensors, GGUF transformers, and low-VRAM-friendly flows.

## Current State
Milestone 1 is complete and pushed. Backend now has model-profile DTOs, admin-gated CRUD/validate/activate endpoints, JSON persistence at `<app_data>/model_profiles.json`, and the LTX model recommendation gate returns `ok` when a valid active official profile exists. No frontend UI, adapter registry, Kijai split loading, or GGUF loading has been implemented yet.

Tracked context lives in this repo now:
- `HANDOFF.md` — this handoff for the next Pi agent.
- `.pi/plans/2026-06-24-ltx-desktop-local-model-profiles-gguf.md` — current implementation plan and stage tracker.
- `.pi/subplans/*.md` — worker planning slices used to build the plan.
- `docs/ltx-offline-research/*.md` — research copied from the old machine so the next PC has it in Git.

Latest commits:
- `85499c0 feat(models): add local model profiles` — Milestone 1 implementation.
- Next commit after this handoff will add tracked docs/context only.

## Files in Flight
- `backend/api_types.py` — model profile DTOs and response models.
- `backend/handlers/model_profiles_handler.py` — profile persistence, CRUD, validation, activation, readiness helper.
- `backend/_routes/model_profiles.py` — admin-gated profile API routes.
- `backend/app_handler.py` — wires `ModelProfilesHandler` before `ModelsHandler` and loads profiles on startup.
- `backend/handlers/models_handler.py` — download recommendation now respects a valid active official profile.
- `backend/tests/test_model_profiles.py` — integration tests for profile API and recommendation readiness.
- `.pi/plans/2026-06-24-ltx-desktop-local-model-profiles-gguf.md` — authoritative roadmap; start at Milestone 2 next.
- `docs/ltx-offline-research/07-official-ltx23-lora-registry.md` — adapter registry source for Milestone 2.

## Changed
- `.gitignore` — now ignores volatile Pi files but allows tracked `.pi/plans/*.md` and `.pi/subplans/*.md`.
- `backend/api_types.py` — added model profile types: paths, profile payloads, patch payload, list/validation/activation responses.
- `backend/state/app_state_types.py` — added `model_profiles` and `active_model_profile_id` to `AppState`.
- `backend/handlers/model_profiles_handler.py` — added profile JSON store and validation.
- `backend/_routes/model_profiles.py` — added `/api/model-profiles` endpoints.
- `backend/app_factory.py` — registers model profile router.
- `backend/app_handler.py` — composes and loads model profile handler.
- `backend/handlers/__init__.py` — exports model profile handler.
- `backend/handlers/models_handler.py` — readiness check for active official profile.
- `backend/tests/test_model_profiles.py` — added 16 integration tests.
- `docs/ltx-offline-research/` — added copied research docs from previous machine.
- `.pi/plans/` and `.pi/subplans/` — added tracked planning artifacts for Pi on the new PC.

## Failed Attempts
- ❌ First subagent could not create new files due scoped tool limits — parent created `backend/handlers/model_profiles_handler.py`, `backend/_routes/model_profiles.py`, and `backend/tests/test_model_profiles.py` manually.
- ❌ `pnpm` was not installed as a direct command — used `npx --yes pnpm@10.30.3 ...` successfully.
- ❌ Initial model-profile tests expected wrong error code for admin guard — fixed to expect `HTTP_403` with message `Admin token required`.
- ❌ Full backend suite rejected `client.patch(...)` because repo forbids `patch(` in tests — switched to `client.request("PATCH", ...)`.
- ❌ Initial `pyright` inferred `list[Unknown]` for `Field(default_factory=list)` and `field(default_factory=list)` — added typed default factories.
- ❌ `gh repo fork Lightricks/LTX-Desktop --remote=false` is unsupported with repo arg — used supported fork command and added `github` remote manually.

## Next Step
Start Milestone 2: implement official LTX-2.3 adapter/LoRA registry and backend status/recommendation endpoints. Use `.pi/plans/2026-06-24-ltx-desktop-local-model-profiles-gguf.md` and `docs/ltx-offline-research/07-official-ltx23-lora-registry.md` first, then add tests before code.
