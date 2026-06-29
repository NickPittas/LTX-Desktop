"""Text encoding cache and API embedding handler."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import ModelSelectionID
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import (
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_ltx_model_spec,
    is_cp_downloaded,
    resolve_model_path,
)
from services.base_video_model_registry import (
    BaseVideoModelRegistryEntry,
    resolve_base_video_model_selection,
)
from state.app_state_types import AppState, TextEncodingResult

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class TextHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)

    def _active_profile_provides_local_encoder(self) -> bool:
        """Check if the active model profile has its own local text encoder."""
        profile_id = self.state.active_model_profile_id
        if profile_id is None:
            return False
        profile = next(
            (p for p in self.state.model_profiles if p.id == profile_id),
            None,
        )
        if profile is None:
            return False
        c = profile.components
        return bool(c.text_encoder_root) and c.text_encoder_format in ("hf_folder", "gguf", "safetensors")

    @with_state_lock
    def _get_cached_prompt(
        self, prompt: str, enhance_prompt: bool, model_identity: str
    ) -> TextEncodingResult | None:
        te = self.state.text_encoder
        if te is None:
            return None
        return te.prompt_cache.get((prompt.strip(), enhance_prompt, model_identity))

    @with_state_lock
    def _cache_prompt(
        self,
        prompt: str,
        enhance_prompt: bool,
        model_identity: str,
        result: TextEncodingResult,
    ) -> None:
        te = self.state.text_encoder
        if te is None:
            return

        max_size = self.state.app_settings.prompt_cache_size
        if max_size <= 0:
            return

        key = (prompt.strip(), enhance_prompt, model_identity)
        if key in te.prompt_cache:
            del te.prompt_cache[key]
        elif len(te.prompt_cache) >= max_size:
            oldest = next(iter(te.prompt_cache))
            del te.prompt_cache[oldest]
        te.prompt_cache[key] = result

    @with_state_lock
    def _set_api_embeddings(self, result: TextEncodingResult | None) -> None:
        if self.state.text_encoder is not None:
            self.state.text_encoder.api_embeddings = result

    def clear_api_embeddings(self) -> None:
        self._set_api_embeddings(None)

    def _resolve_selection_entry(
        self, model_selection: ModelSelectionID | None
    ) -> BaseVideoModelRegistryEntry | None:
        """Resolve a present selection to its registry entry (registry-aware).

        Returns ``None`` when no selection is present. The selection is
        validated/installed-checked by the pipeline handler before text
        encoding runs, so this only resolves the entry (CP-backed and non-CP
        ids alike) via the unified base-video registry.
        """
        if model_selection is None:
            return None
        return resolve_base_video_model_selection(self.models_dir, model_selection)

    def _selected_checkpoint_path(self, model_selection: ModelSelectionID | None) -> str | None:
        """Runtime placement path for a present selection (registry-aware).

        CP-backed selections preserve the existing CP-catalog resolution
        (``resolve_model_path``); non-CP selections (Kijai FP8, QuantStack
        distilled GGUF, official dev safetensors) resolve via the registry's
        filesystem-evidenced ``transformer_path``. The selection is validated
        upstream, so the path is non-None when installed.
        """
        entry = self._resolve_selection_entry(model_selection)
        if entry is None:
            return None
        if entry.download_cp_id is not None:
            # Preserve exact existing behavior for CP-backed selections.
            return str(resolve_model_path(self.models_dir, entry.download_cp_id))
        # Non-CP registry id: actual runtime path (preferred) with canonical
        # placement fallback.
        return entry.transformer_path or entry.expected_absolute_path

    def _effective_model_identity(self, model_selection: ModelSelectionID | None) -> str:
        """Effective base model identity used to namespace prompt/API caches.

        Preference order: selected checkpoint canonical placement path → active
        profile transformer path → downloaded LTX model id. Including this in
        the cache key ensures prompt embeddings never leak across live model
        selections (Step 4).

        Registry-aware: non-CP selections (Kijai/QuantStack/dev) resolve their
        canonical path via the unified base-video registry instead of the
        CP-only ``resolve_model_path`` (which would raise for ids absent from
        the CP catalog). CP-backed selections preserve the existing
        ``resolve_model_path`` identity.
        """
        entry = self._resolve_selection_entry(model_selection)
        if entry is not None:
            if entry.download_cp_id is not None:
                return str(resolve_model_path(self.models_dir, entry.download_cp_id))
            return entry.expected_absolute_path
        profile_id = self.state.active_model_profile_id
        if profile_id is not None:
            profile = next((p for p in self.state.model_profiles if p.id == profile_id), None)
            if profile is not None and profile.components.transformer:
                return profile.components.transformer
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        return model_id or ""

    def should_use_local_encoding(
        self, model_selection: ModelSelectionID | None = None
    ) -> bool:
        """Decide whether to use local text encoding based on availability.

        Text-encoder availability is orthogonal to the base video transformer
        selection, so ``model_selection`` does not change the result; it is
        accepted for call-site symmetry with the other text methods.
        """
        del model_selection
        if self._active_profile_provides_local_encoder():
            return True

        settings = self.state.app_settings.model_copy(deep=True)
        api_available = bool(settings.ltx_api_key)
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        local_available = (
            is_cp_downloaded(self.models_dir, get_ltx_model_spec(model_id).text_encoder_cp)
            if model_id is not None
            else False
        )

        if api_available and local_available:
            return settings.use_local_text_encoder  # setting is tiebreaker for legacy official models only
        return local_available  # use whichever is available

    def prepare_text_encoding(
        self,
        prompt: str,
        enhance_prompt: bool,
        model_selection: ModelSelectionID | None = None,
    ) -> None:
        """Validate settings and prepare text embeddings for a generation run.

        Raises RuntimeError with a prefixed message if text encoding is
        misconfigured, the local encoder is missing, or API encoding fails
        with no local fallback.
        """
        settings = self.state.app_settings.model_copy(deep=True)
        api_available = bool(settings.ltx_api_key)
        profile_local_available = self._active_profile_provides_local_encoder()
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        local_available = profile_local_available or (
            is_cp_downloaded(self.models_dir, get_ltx_model_spec(model_id).text_encoder_cp)
            if model_id is not None
            else False
        )

        if not api_available and not local_available:
            raise RuntimeError(
                "TEXT_ENCODING_NOT_CONFIGURED: To generate videos, you need to configure text encoding. "
                "Either enter an LTX API Key in Settings, or enable the Local Text Encoder."
            )

        use_local = self.should_use_local_encoding(model_selection)
        gemma_root = self.resolve_gemma_root(model_selection)
        model_identity = self._effective_model_identity(model_selection)
        selected_checkpoint_path = self._selected_checkpoint_path(model_selection)
        embeddings = self._prepare_api_embeddings(
            prompt, enhance_prompt, model_identity, selected_checkpoint_path
        )

        if not use_local and embeddings is None and gemma_root is None:
            raise RuntimeError(
                "LTX API text encoding failed and local text encoder is not available. "
                "Please download the text encoder from Settings or check your API key."
            )

    def resolve_gemma_root(
        self, model_selection: ModelSelectionID | None = None
    ) -> str | None:
        # The text-encoder root is independent of the base video transformer
        # selection, so ``model_selection`` does not change the result; accepted
        # for call-site symmetry.
        del model_selection
        if not self.should_use_local_encoding():
            return None
        profile_id = self.state.active_model_profile_id
        if profile_id is not None:
            profile = next((p for p in self.state.model_profiles if p.id == profile_id), None)
            if profile is not None and profile.components.text_encoder_root:
                return profile.components.text_encoder_root
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            return None
        return str(get_existing_cp_path(self.models_dir, get_ltx_model_spec(model_id).text_encoder_cp))

    def _prepare_api_embeddings(
        self,
        prompt: str,
        enhance_prompt: bool,
        model_identity: str,
        selected_checkpoint_path: str | None,
    ) -> TextEncodingResult | None:
        if self.should_use_local_encoding():
            self.clear_api_embeddings()
            return None

        settings = self.state.app_settings.model_copy(deep=True)
        if not settings.ltx_api_key:
            self.clear_api_embeddings()
            return None

        cached = self._get_cached_prompt(prompt, enhance_prompt, model_identity)
        if cached is not None:
            self._set_api_embeddings(cached)
            return cached

        te = self.state.text_encoder
        if te is None:
            return None

        # API embedding checkpoint path follows the live selection when present;
        # otherwise the legacy downloaded distilled model checkpoint is used.
        if selected_checkpoint_path is not None:
            checkpoint_path = selected_checkpoint_path
        else:
            model_id = get_downloaded_ltx_model_id(self.models_dir)
            if model_id is None:
                raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
            checkpoint_path = str(get_existing_cp_path(self.models_dir, get_ltx_model_spec(model_id).model_cp))

        encoded = te.service.encode_via_api(
            prompt=prompt,
            api_key=settings.ltx_api_key,
            checkpoint_path=checkpoint_path,
            enhance_prompt=enhance_prompt,
        )
        if encoded is not None:
            self._cache_prompt(prompt, enhance_prompt, model_identity, encoded)
            self._set_api_embeddings(encoded)
        return encoded
