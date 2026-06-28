import { useCallback, useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type {
  ModelProfileActivateResponse,
  ModelProfileActivationErrorCode,
  ModelProfilePatchPayload,
  ModelProfilePayload,
  ModelProfilesResponse,
  ModelProfileValidationResponse,
} from '../types/model-profile'

export interface ActivationError {
  code: ModelProfileActivationErrorCode
  message: string
}

type ActivationResult =
  | { ok: true; data: ModelProfileActivateResponse }
  | { ok: false; error: ActivationError }

interface ModelProfilesState {
  data: ModelProfilesResponse | null
  isLoading: boolean
  errorMessage: string | null
  refresh: () => Promise<void>
  createProfile: (profile: ModelProfilePayload) => Promise<ModelProfilePayload>
  patchProfile: (profileId: string, patch: ModelProfilePatchPayload) => Promise<ModelProfilePayload>
  deleteProfile: (profileId: string) => Promise<void>
  validateProfile: (profileId: string) => Promise<ModelProfileValidationResponse>
  activateProfile: (profileId: string) => Promise<ModelProfileActivateResponse>
  /** Activates a profile and returns a structured error instead of throwing. */
  activateProfileSafe: (profileId: string) => Promise<ActivationResult>
}

function parseActivationError(error: { code?: string; message: string }): ActivationError {
  const code = error.code ?? 'UNKNOWN'
  const knownCodes: ModelProfileActivationErrorCode[] = [
    'MODEL_PROFILE_ACTIVATION_GENERATION_RUNNING',
    'MODEL_PROFILE_REQUIRED_ARTIFACTS_MISSING',
    'MODEL_PROFILE_CHANGED_DURING_ACTIVATION',
  ]
  if (knownCodes.includes(code as ModelProfileActivationErrorCode)) {
    return { code: code as ModelProfileActivationErrorCode, message: error.message }
  }
  return { code: 'UNKNOWN', message: error.message }
}

function requireOk<T>(result: { ok: true; data: T } | { ok: false; error: { message: string } }): T {
  if (!result.ok) throw new Error(result.error.message)
  return result.data
}

export function useModelProfiles(enabled = true): ModelProfilesState {
  const [data, setData] = useState<ModelProfilesResponse | null>(null)
  const [isLoading, setIsLoading] = useState(enabled)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    const result = await ApiClient.getModelProfiles()
    if (result.ok) {
      setData(result.data)
      setErrorMessage(null)
    } else {
      setErrorMessage(result.error.message)
    }
    setIsLoading(false)
  }, [])

  useEffect(() => {
    if (!enabled) return
    void refresh()
  }, [enabled, refresh])

  const createProfile = useCallback(async (profile: ModelProfilePayload) => {
    const created = requireOk(await ApiClient.createModelProfile(profile))
    await refresh()
    return created
  }, [refresh])

  const patchProfile = useCallback(async (profileId: string, patch: ModelProfilePatchPayload) => {
    const updated = requireOk(await ApiClient.patchModelProfile(profileId, patch))
    await refresh()
    return updated
  }, [refresh])

  const deleteProfile = useCallback(async (profileId: string) => {
    requireOk(await ApiClient.deleteModelProfile(profileId))
    await refresh()
  }, [refresh])

  const validateProfile = useCallback(async (profileId: string) => (
    requireOk(await ApiClient.validateModelProfile(profileId))
  ), [])

  const activateProfile = useCallback(async (profileId: string) => {
    const activated = requireOk(await ApiClient.activateModelProfile(profileId))
    await refresh()
    return activated
  }, [refresh])

  const activateProfileSafe = useCallback(async (profileId: string) => {
    const result = await ApiClient.activateModelProfile(profileId)
    if (result.ok) {
      await refresh()
      return { ok: true as const, data: result.data }
    }
    return { ok: false as const, error: parseActivationError(result.error) }
  }, [refresh])

  return {
    data,
    isLoading,
    errorMessage,
    refresh,
    createProfile,
    patchProfile,
    deleteProfile,
    validateProfile,
    activateProfile,
    activateProfileSafe,
  }
}
