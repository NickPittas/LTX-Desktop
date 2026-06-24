import { useCallback, useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type {
  AdapterPipeline,
  AdapterRecommendationResponse,
  AdapterStatusResponse,
} from '../types/model-profile'

interface OfficialAdaptersState {
  status: AdapterStatusResponse | null
  isLoading: boolean
  errorMessage: string | null
  refresh: () => Promise<void>
  getRecommendation: (pipeline: AdapterPipeline) => Promise<AdapterRecommendationResponse>
}

function requireOk<T>(result: { ok: true; data: T } | { ok: false; error: { message: string } }): T {
  if (!result.ok) throw new Error(result.error.message)
  return result.data
}

export function useOfficialAdapters(pipeline?: AdapterPipeline, enabled = true): OfficialAdaptersState {
  const [status, setStatus] = useState<AdapterStatusResponse | null>(null)
  const [isLoading, setIsLoading] = useState(enabled)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    const result = await ApiClient.getAdapterStatus(pipeline ? { pipeline } : undefined)
    if (result.ok) {
      setStatus(result.data)
      setErrorMessage(null)
    } else {
      setErrorMessage(result.error.message)
    }
    setIsLoading(false)
  }, [pipeline])

  useEffect(() => {
    if (!enabled) return
    void refresh()
  }, [enabled, refresh])

  const getRecommendation = useCallback(async (nextPipeline: AdapterPipeline) => (
    requireOk(await ApiClient.getAdapterRecommendation({ pipeline: nextPipeline }))
  ), [])

  return { status, isLoading, errorMessage, refresh, getRecommendation }
}
