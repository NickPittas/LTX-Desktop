"""Shared base types for state handlers."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

_P = ParamSpec("_P")
_R = TypeVar("_R")
_S = TypeVar("_S", bound="StateHandlerBase")


class StateHandlerBase:
    """Base handler with shared state and lock references."""

    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        self._state = state
        self._lock = lock
        self._config = config

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def models_dir(self) -> Path:
        """Effective models dir: custom from settings, or startup default."""
        custom = self._state.app_settings.models_dir
        return Path(custom) if custom else self._config.default_models_dir

    @property
    def outputs_dir(self) -> Path:
        """Effective generation outputs dir.

        When ``projectAssetsDir`` is set, backend-generated files (EXR/MP4/ProRes/
        image/control temp) are written under a hidden subfolder
        ``<projectAssetsDir>/.ltx-generations`` so the Electron/project copy step
        can organize final assets into per-project folders without source/dest
        collision. Unset -> the startup ``RuntimeConfig.outputs_dir`` fallback.
        """
        custom = self._state.app_settings.project_assets_dir
        if custom:
            out = Path(custom).expanduser().resolve() / ".ltx-generations"
            out.mkdir(parents=True, exist_ok=True)
            return out
        return self._config.outputs_dir


def with_state_lock(
    method: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    @wraps(method)
    def wrapped(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapped
