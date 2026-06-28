import { useCallback, useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type { ModelLibraryScanResponse } from '../types/model-library'

interface UseModelLibraryState {
  catalog: ModelLibraryScanResponse | null
  isLoading: boolean
  errorMessage: string | null
  refresh: () => Promise<void>
}

export function useModelLibrary(enabled = true): UseModelLibraryState {
  const [catalog, setCatalog] = useState<ModelLibraryScanResponse | null>(null)
  const [isLoading, setIsLoading] = useState(enabled)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    const result = await ApiClient.getModelCatalog()
    if (result.ok) {
      setCatalog(result.data)
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

  return { catalog, isLoading, errorMessage, refresh }
}
