"""Model profile CRUD and validation logic."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from _routes._errors import HTTPError
from api_types import (
    CURRENT_MODEL_PROFILE_SCHEMA_VERSION,
    ModelProfileActivateResponse,
    ModelProfilePatchPayload,
    ModelProfilePayload,
    ModelProfileValidationIssuePayload,
    ModelProfileValidationResponse,
    ModelProfilesResponse,
)
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import UPSAMPLER_CP_ID, resolve_model_path
from services.model_resolver import ProfileCapabilityResult, resolve_profile_capabilities
from services.model_scanner import scan_models
from state.app_state_types import ApiGeneration, GenerationRunning, GpuGeneration

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState


_PROFILES_FILE = "model_profiles.json"
_SAFE_TENSORS_FIELDS = (
    "upsampler",
    "text_projection",
    "embeddings_connector",
    "video_vae",
    "audio_vae",
    "vocoder",
    "ic_lora_union",
    "ic_lora_motion_track",
    "ic_lora_ingredients",
    "ic_lora_hdr",
    "ic_lora_hdr_scene_embeddings",
    "ic_lora_lipdub",
    "ic_lora_in_outpainting",
    "depth_processor",
    "pose_processor",
    "person_detector",
)
_PATH_FIELDS = (
    "transformer",
    "upsampler",
    "text_encoder_root",
    "text_projection",
    "embeddings_connector",
    "video_vae",
    "audio_vae",
    "vocoder",
    *_SAFE_TENSORS_FIELDS[6:],
)


def _activation_fingerprint(profile: ModelProfilePayload) -> tuple[object, ...]:
    """Stable tuple of profile fields that affect activation safety.

    Captures the fields the resolver consults for base-diffusion resolution so
    a concurrent patch between the pre-scan and the commit is detected.
    """
    c = profile.components
    return (
        profile.id,
        profile.validation_status,
        c.transformer,
        c.transformer_format,
        c.transformer_quantization,
    )


def _base_diffusion_resolvable(capabilities: ProfileCapabilityResult) -> bool:
    """Phase 5 activation gate: the base diffusion model (transformer) must be
    explicitly named by the profile AND scanner-recognized.

    Catalog fallback alone is NOT accepted — ``normal_status == "supported"``
    can be true even when ``components.transformer`` is empty (catalog
    installed/duplicate fills in). Instead the resolved base artifact must:

    - originate from the profile explicit path (``source == "profile"``), and
    - have a scanned-and-usable status (``available`` or ``duplicate``).

    This admits a wrong-folder transformer the profile explicitly points at
    (the resolver marks it ``available`` / source ``profile``) and rejects a
    bare catalog-installed artifact when the profile omits the path, an
    unverified profile-only path, and any non-usable status.
    """
    base = next(
        (a for a in capabilities.artifacts if a.component_role == "base_diffusion_model"),
        None,
    )
    if base is None:
        return False
    return base.source == "profile" and base.status in ("available", "duplicate")


def _migrate_raw_profile_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply schema migrations to a raw profile dict before Pydantic validation.

    Phase 1: absent ``schema_version`` is treated as legacy and normalized to
    the current version. This preserves ``extra="forbid"`` on
    :class:`~api_types.ModelProfilePayload` by ensuring new fields are present
    in the dict before validation rather than weakening the schema.

    This function is a no-op for already-current profiles and does NOT trigger
    a file write — the persisted ``model_profiles.json`` is only updated on
    explicit save/patch/create (or the existing blank-ID repair path).
    """
    if "schema_version" not in raw:
        return {**raw, "schema_version": CURRENT_MODEL_PROFILE_SCHEMA_VERSION}
    return raw


# Basename of the canonical 2x spatial upscaler artifact. Legacy/wizard-bug
# profiles stored a stale root-level path; safe canonicalization rewrites
# those under the effective models root when the canonical artifact exists.
_UPSAMPLER_BASENAME = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"


def _canonicalize_upsampler_path(path_str: str, models_dir: Path) -> str:
    """Return a safe canonical upscaler path when a stale profile path is fixable.

    Always normalizes doubled slashes / stray separators via :class:`Path`
    (POSIX collapses consecutive slashes). Only rewrites to canonical when ALL
    of the following hold:

    - the path basename is ``ltx-2.3-spatial-upscaler-x2-1.0.safetensors``;
    - the normalized path is *under* ``models_dir`` (arbitrary external paths
      are never rewritten);
    - the canonical path (``latent_upscale_models/...`` under ``models_dir``)
      actually exists on disk;
    - the normalized path differs from canonical (i.e. current path is stale
      or missing — when the current path is already the canonical one, this is
      a no-op).

    Scanner is read-only: this never moves, deletes, or downloads model
    files. It only rewrites the in-memory profile string. The caller is
    responsible for persisting the change (load/save/patch/create).
    """
    normalized = str(Path(path_str))
    try:
        Path(normalized).relative_to(models_dir)
    except ValueError:
        # External path — never rewrite, just return normalized form.
        return normalized

    if Path(normalized).name != _UPSAMPLER_BASENAME:
        return normalized

    canonical_str = str(resolve_model_path(models_dir, UPSAMPLER_CP_ID))
    if normalized == canonical_str:
        return normalized

    if not Path(canonical_str).exists():
        # No canonical target to rewrite to; preserve current (normalized).
        return normalized

    # Current path is stale/missing or differs from a canonical artifact that
    # exists under models_dir — safe to rewrite.
    return canonical_str


def _canonicalize_profile_paths(profile: ModelProfilePayload, models_dir: Path) -> bool:
    """In-place canonicalize known stale component paths.

    Returns ``True`` when any path was rewritten (caller persists).
    """
    changed = False
    upsampler = profile.components.upsampler
    if upsampler:
        fixed = _canonicalize_upsampler_path(upsampler, models_dir)
        if fixed != upsampler:
            profile.components.upsampler = fixed
            changed = True
    return changed


class ModelProfilesHandler(StateHandlerBase):
    """Persist, list, create, patch, delete, validate, and activate profiles."""

    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)
        self._profiles_path = config.app_data_dir / _PROFILES_FILE

    def load_profiles(self) -> None:
        """Load profiles from disk into app state."""
        path = self._profiles_path
        if not path.exists():
            self.state.model_profiles = []
            self.state.active_model_profile_id = None
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            migrated = [_migrate_raw_profile_dict(p) for p in raw.get("profiles", [])]
            profiles = [ModelProfilePayload(**profile) for profile in migrated]
            active_id = raw.get("active_model_profile_id")
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            profiles = []
            active_id = None

        # Repair blank-ID profiles by assigning generated IDs
        repaired = False
        existing_ids = {p.id for p in profiles}
        for profile in profiles:
            if not profile.id.strip():
                profile.id = uuid.uuid4().hex
                while profile.id in existing_ids:
                    profile.id = uuid.uuid4().hex
                existing_ids.add(profile.id)
                repaired = True

        # Safe canonicalization of stale component paths under models_dir
        # (e.g. legacy root-level upsampler when canonical exists). Persisted
        # like the blank-ID repair above so future loads are idempotent.
        models_dir = self.models_dir
        for profile in profiles:
            if _canonicalize_profile_paths(profile, models_dir):
                repaired = True

        self.state.model_profiles = profiles
        if active_id and active_id.strip() and active_id in existing_ids:
            self.state.active_model_profile_id = active_id
        else:
            self.state.active_model_profile_id = None
            if active_id is not None:
                repaired = True

        if repaired:
            self.save_profiles()

    def save_profiles(self) -> None:
        """Persist current model profiles to disk."""
        self._profiles_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_model_profile_id": self.state.active_model_profile_id,
            "profiles": [profile.model_dump() for profile in self.state.model_profiles],
        }
        self._profiles_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @with_state_lock
    def list_profiles(self) -> ModelProfilesResponse:
        return ModelProfilesResponse(
            active_model_profile_id=self.state.active_model_profile_id,
            profiles=list(self.state.model_profiles),
        )

    @with_state_lock
    def create_profile(self, payload: ModelProfilePayload) -> ModelProfilePayload:
        if not payload.id.strip():
            payload.id = uuid.uuid4().hex
            while any(profile.id == payload.id for profile in self.state.model_profiles):
                payload.id = uuid.uuid4().hex
        elif any(profile.id == payload.id for profile in self.state.model_profiles):
            raise HTTPError(409, "PROFILE_ID_ALREADY_EXISTS")
        _canonicalize_profile_paths(payload, self.models_dir)
        self.state.model_profiles.append(payload)
        self.save_profiles()
        return payload

    @with_state_lock
    def patch_profile(self, profile_id: str, patch: ModelProfilePatchPayload) -> ModelProfilePayload:
        for index, profile in enumerate(self.state.model_profiles):
            if profile.id == profile_id:
                updated_payload = profile.model_dump()
                updated_payload.update(patch.model_dump(exclude_unset=True))
                updated = ModelProfilePayload(**updated_payload)
                _canonicalize_profile_paths(updated, self.models_dir)
                self.state.model_profiles[index] = updated
                self.save_profiles()
                return updated
        raise HTTPError(404, "PROFILE_NOT_FOUND")

    @with_state_lock
    def delete_profile(self, profile_id: str) -> None:
        for index, profile in enumerate(self.state.model_profiles):
            if profile.id == profile_id:
                self.state.model_profiles.pop(index)
                if self.state.active_model_profile_id == profile_id:
                    self.state.active_model_profile_id = None
                self.save_profiles()
                return
        raise HTTPError(404, "PROFILE_NOT_FOUND")

    def activate_profile(self, profile_id: str) -> ModelProfileActivateResponse:
        """Activate a profile with lock-safe, resolver-backed safety checks.

        Three-phase locking so no scan/disk-IO happens under the state lock:

        1. Brief lock: reject if generation running; find + deep-copy the
           target profile; snapshot ``models_dir`` and an activation fingerprint.
        2. Outside lock: scan ``models_dir`` and resolve capabilities; reject
           if the required base diffusion model is not scanner-resolvable.
        3. Re-lock: reject if generation is now running, or the profile /
           models_dir changed between phases; otherwise commit. Activation may
           activate a ``candidate`` profile but never promotes it to
           ``validated`` (live validation is Phase 8).
        """
        # Phase 1 — brief lock to snapshot.
        with self._lock:
            if self._is_generation_running():
                raise HTTPError(409, "MODEL_PROFILE_ACTIVATION_GENERATION_RUNNING")
            profile = self._find_profile(profile_id)  # raises 404 if missing
            profile_snapshot = profile.model_copy(deep=True)
            fingerprint = _activation_fingerprint(profile)
            models_dir = self.models_dir

        # Phase 2 — scan + resolve outside the lock (disk IO).
        catalog = scan_models(models_dir)
        capabilities = resolve_profile_capabilities(profile_snapshot, catalog)
        if not _base_diffusion_resolvable(capabilities):
            raise HTTPError(409, "MODEL_PROFILE_REQUIRED_ARTIFACTS_MISSING")

        # Phase 3 — re-lock to recheck and commit.
        with self._lock:
            if self._is_generation_running():
                raise HTTPError(409, "MODEL_PROFILE_ACTIVATION_GENERATION_RUNNING")
            current = next(
                (p for p in self.state.model_profiles if p.id == profile_id), None
            )
            if (
                current is None
                or _activation_fingerprint(current) != fingerprint
                or self.models_dir != models_dir
            ):
                raise HTTPError(409, "MODEL_PROFILE_CHANGED_DURING_ACTIVATION")
            self.state.active_model_profile_id = profile_id
            self.save_profiles()
            return ModelProfileActivateResponse(status="ok", active_model_profile_id=profile_id)

    def _is_generation_running(self) -> bool:
        """True when a generation is currently running.

        Reads shared state; the caller is responsible for holding the state
        lock so the observation is consistent.
        """
        active = self.state.active_generation
        if active is None:
            return False
        match active:
            case GpuGeneration(state=generation) if self.state.gpu_slot is not None:
                return isinstance(generation, GenerationRunning)
            case ApiGeneration(state=generation):
                return isinstance(generation, GenerationRunning)
            case _:
                return False

    @with_state_lock
    def validate_profile_by_id(self, profile_id: str) -> ModelProfileValidationResponse:
        return self.validate_profile(self._find_profile(profile_id))

    def validate_profile(self, profile: ModelProfilePayload) -> ModelProfileValidationResponse:
        """Check every configured path exists and key extensions match format."""
        issues: list[ModelProfileValidationIssuePayload] = []
        components = profile.components

        self._add_transformer_issue(issues, components.transformer, components.transformer_format)

        for field_name in _PATH_FIELDS:
            self._add_missing_path_issue(issues, f"components.{field_name}", getattr(components, field_name))
        for field_name in _SAFE_TENSORS_FIELDS:
            self._add_ext_issue(issues, f"components.{field_name}", getattr(components, field_name), ".safetensors")

        self._add_text_encoder_issues(
            issues,
            components.text_encoder_root,
            components.text_encoder_format,
        )
        for adapter_id, adapter_path in components.official_adapters.items():
            self._add_missing_path_issue(issues, f"components.official_adapters.{adapter_id}", adapter_path)

        return ModelProfileValidationResponse(valid=not issues, issues=issues)

    def has_valid_active_official_profile(self) -> bool:
        """Return true when active official profile validates."""
        profile = self._find_active_profile()
        if profile is None or profile.source != "official":
            return False
        # ponytail: API-key readiness deferred until text encoder profiles are wired.
        return self.validate_profile(profile).valid

    def has_valid_active_profile(self) -> bool:
        """Return true when any active profile validates."""
        profile = self._find_active_profile()
        return profile is not None and self.validate_profile(profile).valid

    def _find_profile(self, profile_id: str) -> ModelProfilePayload:
        for profile in self.state.model_profiles:
            if profile.id == profile_id:
                return profile
        raise HTTPError(404, "PROFILE_NOT_FOUND")

    def _find_active_profile(self) -> ModelProfilePayload | None:
        profile_id = self.state.active_model_profile_id
        if profile_id is None:
            return None
        return next((profile for profile in self.state.model_profiles if profile.id == profile_id), None)

    def _add_transformer_issue(
        self,
        issues: list[ModelProfileValidationIssuePayload],
        path_str: str | None,
        transformer_format: str,
    ) -> None:
        suffix = ".gguf" if transformer_format == "gguf" else ".safetensors"
        self._add_ext_issue(issues, "components.transformer", path_str, suffix)

    def _add_text_encoder_issues(
        self,
        issues: list[ModelProfileValidationIssuePayload],
        path_str: str | None,
        text_encoder_format: str,
    ) -> None:
        if not path_str:
            return
        if text_encoder_format == "hf_folder" and not Path(path_str).is_dir():
            issues.append(
                ModelProfileValidationIssuePayload(
                    field="components.text_encoder_root",
                    issue=f"Expected directory: {path_str}",
                )
            )
        elif text_encoder_format == "safetensors":
            self._add_ext_issue(issues, "components.text_encoder_root", path_str, ".safetensors")
        elif text_encoder_format == "gguf":
            self._add_ext_issue(issues, "components.text_encoder_root", path_str, ".gguf")

    def _add_missing_path_issue(
        self,
        issues: list[ModelProfileValidationIssuePayload],
        field_name: str,
        path_str: str | None,
    ) -> None:
        if path_str and not Path(path_str).exists():
            issues.append(
                ModelProfileValidationIssuePayload(
                    field=field_name,
                    issue=f"Path does not exist: {path_str}",
                )
            )

    def _add_ext_issue(
        self,
        issues: list[ModelProfileValidationIssuePayload],
        field_name: str,
        path_str: str | None,
        expected_suffix: str,
    ) -> None:
        if path_str and not path_str.lower().endswith(expected_suffix):
            issues.append(
                ModelProfileValidationIssuePayload(
                    field=field_name,
                    issue=f"Expected {expected_suffix} extension: {path_str}",
                )
            )
