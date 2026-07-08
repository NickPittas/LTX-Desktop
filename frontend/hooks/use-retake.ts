import { useCallback, useRef, useState } from 'react'
import { ApiClient, type ApiRequestBodyOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import type { OutputFormat } from '../lib/output-formats'
import type { ModelSelectionID } from '../lib/model-selection'
import { getPhaseMessage } from './use-generation'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'

export interface RetakeSubmitParams {
  videoPath: string
  startTime: number
  duration: number
  prompt: string
  mode: RetakeMode
  outputFormat?: OutputFormat
  modelSelection?: ModelSelectionID | null
}

export interface RetakeResult {
  videoPath: string
  proxyPath: string | null
  generationElapsedSeconds?: number
}

type RetakeBody = ApiRequestBodyOf<'retake'>

interface UseRetakeState {
  isRetaking: boolean
  retakeStatus: string
  retakeError: string | null
  result: RetakeResult | null
  phaseDetail: string | null
  workloadMode: string | null
}

export function useRetake() {
  const [state, setState] = useState<UseRetakeState>({
    isRetaking: false,
    retakeStatus: '',
    retakeError: null,
    result: null,
    phaseDetail: null,
    workloadMode: null,
  })

  const onCompleteRef = useRef<((result: RetakeResult) => void) | undefined>()

  const submitRetake = useCallback(async (params: RetakeSubmitParams, onComplete?: (result: RetakeResult) => void) => {
    if (!params.videoPath) return

    onCompleteRef.current = onComplete

    setState({
      isRetaking: true,
      retakeStatus: 'Generating',
      retakeError: null,
      result: null,
      phaseDetail: null,
      workloadMode: null,
    })

    const body: Record<string, unknown> = {
      video_path: params.videoPath,
      start_time: params.startTime,
      duration: params.duration,
      prompt: params.prompt,
      mode: params.mode,
    }
    if (params.outputFormat && params.outputFormat !== 'mp4') {
      body.output_format = params.outputFormat
    }
    if (params.modelSelection !== undefined && params.modelSelection !== null) {
      body.model_selection = params.modelSelection
    }

    // Poll the shared generation-progress endpoint for live status + elapsed
    // while the synchronous retake request is in flight (same contract as
    // use-ic-lora/use-generation).
    const startWall = Date.now()
    let latestElapsed: number | null = null
    let shouldApplyPollingUpdates = true
    const pollProgress = async () => {
      if (!shouldApplyPollingUpdates) return
      const r = await ApiClient.getGenerationProgress()
      if (!r.ok || !shouldApplyPollingUpdates) return
      if (r.data.elapsedSeconds != null) latestElapsed = r.data.elapsedSeconds
      setState(prev => ({
        ...prev,
        retakeStatus: getPhaseMessage(r.data.phase),
        phaseDetail: r.data.phaseDetail ?? null,
        workloadMode: r.data.workloadMode ?? null,
      }))
    }
    const progressInterval = setInterval(pollProgress, 500)

    const result = await ApiClient.retake(body as RetakeBody)
    shouldApplyPollingUpdates = false
    clearInterval(progressInterval)

    if (!result.ok) {
      logger.error(`Retake error: ${result.error.message}`)
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: result.error.message,
        result: null,
        phaseDetail: null,
        workloadMode: null,
      })
      return
    }

    const payload = result.data

    if (payload.status === 'cancelled') {
      setState({
        isRetaking: false,
        retakeStatus: 'Cancelled',
        retakeError: null,
        result: null,
        phaseDetail: null,
        workloadMode: null,
      })
      return
    }

    if ('video_path' in payload) {
      const finalElapsed = Math.max(latestElapsed ?? 0, (Date.now() - startWall) / 1000)
      const res: RetakeResult = {
        videoPath: payload.video_path,
        proxyPath: payload.proxy_path ?? null,
        generationElapsedSeconds: finalElapsed,
      }
      // Fire onComplete before local setState — runs ProjectContext mutations
      // even if GenSpace has unmounted (Bug A fix)
      onCompleteRef.current?.(res)
      onCompleteRef.current = undefined
      setState({
        isRetaking: false,
        retakeStatus: 'Retake complete!',
        retakeError: null,
      result: res,
      phaseDetail: null,
      workloadMode: null,
    })
      return
    }

    logger.error(`Retake completed without local video payload: ${JSON.stringify(payload.result)}`)
    const errorMsg = 'Retake completed but no local video file was returned'
    setState({
      isRetaking: false,
      retakeStatus: '',
      retakeError: errorMsg,
      result: null,
      phaseDetail: null,
      workloadMode: null,
    })
  }, [])

  const resetRetake = useCallback(() => {
    setState({
      isRetaking: false,
      retakeStatus: '',
      retakeError: null,
      result: null,
      phaseDetail: null,
      workloadMode: null,
    })
  }, [])

  return {
    submitRetake,
    resetRetake,
    isRetaking: state.isRetaking,
    retakeStatus: state.retakeStatus,
    retakeError: state.retakeError,
    retakeResult: state.result,
    retakePhaseDetail: state.phaseDetail,
    retakeWorkloadMode: state.workloadMode,
  }
}
