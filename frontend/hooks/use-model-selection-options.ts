import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type { ModelSelectionOption, ModelSelectionWorkflow } from '../lib/model-selection'

interface UseModelSelectionOptionsState {
  options: ModelSelectionOption[]
  isLoading: boolean
  errorMessage: string | null
  refresh: () => Promise<ModelSelectionOption[]>
}

export function useModelSelectionOptions(
  workflow: ModelSelectionWorkflow | null,
): UseModelSelectionOptionsState {
  const [options, setOptions] = useState<ModelSelectionOption[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const latestRequestRef = useRef<AbortController | null>(null)

  const fetchOptions = useCallback(async (targetWorkflow: ModelSelectionWorkflow): Promise<ModelSelectionOption[]> => {
    latestRequestRef.current?.abort()
    const abortController = new AbortController()
    latestRequestRef.current = abortController

    setIsLoading(true)
    setErrorMessage(null)

    const result = await ApiClient.getModelSelectionOptions({ workflow: targetWorkflow }, {
      signal: abortController.signal,
    })

    if (abortController.signal.aborted) {
      return []
    }

    if (latestRequestRef.current !== abortController) {
      return result.ok ? result.data.options : []
    }

    if (result.ok) {
      setOptions(result.data.options)
      setIsLoading(false)
      setErrorMessage(null)
      return result.data.options
    }

    setOptions([])
    setIsLoading(false)
    setErrorMessage(result.error.message)
    return []
  }, [])

  const refresh = useCallback(async (): Promise<ModelSelectionOption[]> => {
    if (!workflow) {
      setOptions([])
      setIsLoading(false)
      setErrorMessage(null)
      return []
    }
    return fetchOptions(workflow)
  }, [workflow, fetchOptions])

  useEffect(() => {
    if (!workflow) {
      latestRequestRef.current?.abort()
      latestRequestRef.current = null
      setOptions([])
      setIsLoading(false)
      setErrorMessage(null)
      return
    }

    void fetchOptions(workflow)

    return () => {
      latestRequestRef.current?.abort()
      latestRequestRef.current = null
    }
  }, [workflow, fetchOptions])

  return { options, isLoading, errorMessage, refresh }
}
