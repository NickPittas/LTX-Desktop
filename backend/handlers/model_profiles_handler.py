"""Model profile CRUD and validation logic."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import (
    ModelProfileActivateResponse,
    ModelProfilePatchPayload,
    ModelProfilePayload,
    ModelProfileValidationIssuePayload,
    ModelProfileValidationResponse,
    ModelProfilesResponse,
)
from handlers.base import StateHandlerBase, with_state_lock

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
            profiles = [ModelProfilePayload(**profile) for profile in raw.get("profiles", [])]
            active_id = raw.get("active_model_profile_id")
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            profiles = []
            active_id = None

        profile_ids = {profile.id for profile in profiles}
        self.state.model_profiles = profiles
        self.state.active_model_profile_id = active_id if active_id in profile_ids else None

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
        if any(profile.id == payload.id for profile in self.state.model_profiles):
            raise HTTPError(409, "PROFILE_ID_ALREADY_EXISTS")
        self.state.model_profiles.append(payload)
        self.save_profiles()
        return payload

    @with_state_lock
    def patch_profile(self, profile_id: str, patch: ModelProfilePatchPayload) -> ModelProfilePayload:
        for index, profile in enumerate(self.state.model_profiles):
            if profile.id == profile_id:
                updated = profile.model_copy(update=patch.model_dump(exclude_unset=True))
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

    @with_state_lock
    def activate_profile(self, profile_id: str) -> ModelProfileActivateResponse:
        profile = self._find_profile(profile_id)
        validation = self.validate_profile(profile)
        if not validation.valid:
            raise HTTPError(409, "MODEL_PROFILE_INVALID")
        self.state.active_model_profile_id = profile_id
        self.save_profiles()
        return ModelProfileActivateResponse(status="ok", active_model_profile_id=profile_id)

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
