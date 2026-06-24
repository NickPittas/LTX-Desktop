import { useCallback, useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type {
  ModelProfileActivateResponse,
  ModelProfilePatchPayload,
  ModelProfilePayload,
  ModelProfilesResponse,
  ModelProfileValidationResponse,
} from '../types/model-profile'

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
  }
}
