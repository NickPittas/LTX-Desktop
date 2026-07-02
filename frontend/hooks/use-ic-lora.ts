import { useCallback, useRef, useState } from 'react'
import { ApiClient, type ApiRequestBodyOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import type { OutputFormat } from '../lib/output-formats'
import type { ModelSelectionID } from '../lib/model-selection'
import { getPhaseMessage } from './use-generation'

export type IcLoraConditioningType = 'canny' | 'depth'

export interface IcLoraSubmitParams {
  videoPath: string | null
  conditioningType: IcLoraConditioningType | null
  conditioningStrength: number
  prompt: string
  adapterId?: string | null
  maskPath?: string | null
  maskGrowPx?: number
  laplacianBlendGrow?: number
  finalMaskBlurPx?: number
  loraStrength?: number
  frameRate?: number
  width?: number
  height?: number
  numFrames?: number
  images?: { path: string; frame?: number; strength?: number }[]
  outputFormat?: OutputFormat
  modelSelection?: ModelSelectionID | null
}

export interface IcLoraResult {
  videoPath: string
  proxyPath: string | null
  generationElapsedSeconds?: number
}

interface UseIcLoraState {
  isGenerating: boolean
  status: string
  error: string | null
  result: IcLoraResult | null
  // Live progress/metrics (null when unavailable); mirror use-generation.
  progress: number
  elapsedSeconds: number | null
  estimatedRemainingSeconds: number | null
  vramUsedMb: number | null
  vramTotalMb: number | null
  ramUsedMb: number | null
  ramTotalMb: number | null
  gpuUtilPct: number | null
  cpuUtilPct: number | null
}

type GenerateIcLoraBody = ApiRequestBodyOf<'generateIcLora'>

const NULL_METRICS = {
  progress: 0,
  elapsedSeconds: null as number | null,
  estimatedRemainingSeconds: null as number | null,
  vramUsedMb: null as number | null,
  vramTotalMb: null as number | null,
  ramUsedMb: null as number | null,
  ramTotalMb: null as number | null,
  gpuUtilPct: null as number | null,
  cpuUtilPct: null as number | null,
}

export function useIcLora() {
  const [state, setState] = useState<UseIcLoraState>({
    isGenerating: false,
    status: '',
    error: null,
    result: null,
    ...NULL_METRICS,
  })

  const onCompleteRef = useRef<((result: IcLoraResult) => void) | undefined>()

  const submitIcLora = useCallback(async (params: IcLoraSubmitParams, onComplete?: (result: IcLoraResult) => void) => {
    onCompleteRef.current = onComplete

    setState({
      isGenerating: true,
      status: 'Generating',
      error: null,
      result: null,
      ...NULL_METRICS,
    })

    const body: Record<string, unknown> = {
      conditioning_strength: params.conditioningStrength,
      lora_strength: params.loraStrength ?? undefined,
      prompt: params.prompt,
      frame_rate: params.frameRate ?? 24,
    }
    if (params.outputFormat && params.outputFormat !== 'mp4') {
      body.output_format = params.outputFormat
    }
    if (params.videoPath) {
      body.video_path = params.videoPath
    }
    if (params.conditioningType !== null) {
      body.conditioning_type = params.conditioningType
    }
    if (params.adapterId) {
      body.adapter_id = params.adapterId
    }
    if (params.maskPath) {
      body.mask_path = params.maskPath
    }
    body.mask_grow_px = params.maskGrowPx ?? 30
    body.laplacian_blend_grow = params.laplacianBlendGrow ?? 12
    body.final_mask_blur_px = params.finalMaskBlurPx ?? 6
    if (params.width !== undefined) body.width = params.width
    if (params.height !== undefined) body.height = params.height
    if (params.numFrames !== undefined) body.num_frames = params.numFrames
    if (params.images && params.images.length > 0) {
      body.images = params.images
    }
    if (params.modelSelection !== undefined && params.modelSelection !== null) {
      body.model_selection = params.modelSelection
    }

    // Poll the shared generation-progress endpoint for live metrics while the
    // synchronous IC-LoRA request is in flight (same contract as use-generation).
    const startWall = Date.now()
    let latestElapsed: number | null = null
    let shouldApplyPollingUpdates = true
    const pollProgress = async () => {
      if (!shouldApplyPollingUpdates) return
      const r = await ApiClient.getGenerationProgress()
      if (!r.ok || !shouldApplyPollingUpdates) return
      const data = r.data
      let displayProgress = data.progress
      let status = getPhaseMessage(data.phase)
      if (data.phase === 'complete' || data.status === 'complete') {
        displayProgress = 95
        status = 'Finalizing...'
      }
      if (data.elapsedSeconds != null) latestElapsed = data.elapsedSeconds
      setState(prev => ({
        ...prev,
        progress: displayProgress,
        status,
        elapsedSeconds: data.elapsedSeconds ?? null,
        estimatedRemainingSeconds: data.estimatedRemainingSeconds ?? null,
        vramUsedMb: data.vramUsedMb ?? null,
        vramTotalMb: data.vramTotalMb ?? null,
        ramUsedMb: data.ramUsedMb ?? null,
        ramTotalMb: data.ramTotalMb ?? null,
        gpuUtilPct: data.gpuUtilPct ?? null,
        cpuUtilPct: data.cpuUtilPct ?? null,
      }))
    }
    const progressInterval = setInterval(pollProgress, 500)

    try {
      const result = await ApiClient.generateIcLora(body as GenerateIcLoraBody)
      shouldApplyPollingUpdates = false
      if (!result.ok) {
        logger.error(`IC-LoRA error: ${result.error.message}`)
        setState({
          isGenerating: false,
          status: '',
          error: result.error.message,
          result: null,
          ...NULL_METRICS,
        })
        return
      }

      const payload = result.data
      if (payload.status === 'cancelled') {
        setState({
          isGenerating: false,
          status: 'Cancelled',
          error: null,
          result: null,
          ...NULL_METRICS,
        })
        return
      }

      if (payload.status === 'complete') {
        const finalElapsed = Math.max(latestElapsed ?? 0, (Date.now() - startWall) / 1000)
        const res: IcLoraResult = {
          videoPath: payload.video_path,
          proxyPath: payload.proxy_path ?? null,
          generationElapsedSeconds: finalElapsed,
        }
        // Fire onComplete before local setState — runs ProjectContext mutations
        // even if GenSpace has unmounted (Bug A fix)
        onCompleteRef.current?.(res)
        onCompleteRef.current = undefined
        setState(prev => ({
          ...prev,
          isGenerating: false,
          status: 'Generation complete!',
          error: null,
          result: res,
        }))
        return
      }
    } finally {
      shouldApplyPollingUpdates = false
      clearInterval(progressInterval)
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      status: '',
      error: null,
      result: null,
      ...NULL_METRICS,
    })
  }, [])

  return {
    submitIcLora,
    resetIcLora: reset,
    isIcLoraGenerating: state.isGenerating,
    icLoraStatus: state.status,
    icLoraError: state.error,
    icLoraResult: state.result,
    icLoraProgress: state.progress,
    icLoraElapsedSeconds: state.elapsedSeconds,
    icLoraEstimatedRemainingSeconds: state.estimatedRemainingSeconds,
    icLoraVramUsedMb: state.vramUsedMb,
    icLoraVramTotalMb: state.vramTotalMb,
    icLoraGpuUtilPct: state.gpuUtilPct,
    icLoraRamUsedMb: state.ramUsedMb,
    icLoraRamTotalMb: state.ramTotalMb,
    icLoraCpuUtilPct: state.cpuUtilPct,
  }
}
