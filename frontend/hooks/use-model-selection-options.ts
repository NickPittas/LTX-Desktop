import { useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type { ModelSelectionOption, ModelSelectionWorkflow } from '../lib/model-selection'

interface UseModelSelectionOptionsState {
  options: ModelSelectionOption[]
  isLoading: boolean
  errorMessage: string | null
}

const EMPTY_STATE: UseModelSelectionOptionsState = {
  options: [],
  isLoading: false,
  errorMessage: null,
}

export function useModelSelectionOptions(
  workflow: ModelSelectionWorkflow | null,
): UseModelSelectionOptionsState {
  const [state, setState] = useState<UseModelSelectionOptionsState>(EMPTY_STATE)

  useEffect(() => {
    if (!workflow) {
      setState(EMPTY_STATE)
      return
    }

    const abortController = new AbortController()
    let isActive = true

    setState((prev) => ({ ...prev, isLoading: true, errorMessage: null }))

    void (async () => {
      const result = await ApiClient.getModelSelectionOptions({ workflow }, {
        signal: abortController.signal,
      })
      if (!isActive) return

      if (result.ok) {
        setState({
          options: result.data.options,
          isLoading: false,
          errorMessage: null,
        })
        return
      }

      setState({
        options: [],
        isLoading: false,
        errorMessage: result.error.message,
      })
    })()

    return () => {
      isActive = false
      abortController.abort()
    }
  }, [workflow])

  return state
}
