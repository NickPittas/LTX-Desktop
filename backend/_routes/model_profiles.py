"""Route handlers for model profile APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from _routes._admin_guard import guard_admin_permission
from api_types import (
    ModelProfileActivateResponse,
    ModelProfilePatchPayload,
    ModelProfilePayload,
    ModelProfileValidationResponse,
    ModelProfilesResponse,
    StatusResponse,
)
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["model-profiles"])


@router.get("/model-profiles", response_model=ModelProfilesResponse)
def route_list_profiles(
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> ModelProfilesResponse:
    guard_admin_permission(request)
    return handler.model_profiles.list_profiles()


@router.post("/model-profiles", response_model=ModelProfilePayload)
def route_create_profile(
    req: ModelProfilePayload,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> ModelProfilePayload:
    guard_admin_permission(request)
    return handler.model_profiles.create_profile(req)


@router.patch("/model-profiles/{profile_id}", response_model=ModelProfilePayload)
def route_patch_profile(
    profile_id: str,
    req: ModelProfilePatchPayload,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> ModelProfilePayload:
    guard_admin_permission(request)
    return handler.model_profiles.patch_profile(profile_id, req)


@router.delete("/model-profiles/{profile_id}", response_model=StatusResponse)
def route_delete_profile(
    profile_id: str,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    guard_admin_permission(request)
    handler.model_profiles.delete_profile(profile_id)
    return StatusResponse(status="ok")


@router.post("/model-profiles/{profile_id}/validate", response_model=ModelProfileValidationResponse)
def route_validate_profile(
    profile_id: str,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> ModelProfileValidationResponse:
    guard_admin_permission(request)
    return handler.model_profiles.validate_profile_by_id(profile_id)


@router.post("/model-profiles/{profile_id}/activate", response_model=ModelProfileActivateResponse)
def route_activate_profile(
    profile_id: str,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> ModelProfileActivateResponse:
    guard_admin_permission(request)
    return handler.model_profiles.activate_profile(profile_id)
