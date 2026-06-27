import { useCallback, useRef, useState } from 'react'
import { ApiClient, type ApiRequestBodyOf } from '../lib/api-client'
import { logger } from '../lib/logger'

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
}

export interface IcLoraResult {
  videoPath: string
}

interface UseIcLoraState {
  isGenerating: boolean
  status: string
  error: string | null
  result: IcLoraResult | null
}

type GenerateIcLoraBody = ApiRequestBodyOf<'generateIcLora'>

export function useIcLora() {
  const [state, setState] = useState<UseIcLoraState>({
    isGenerating: false,
    status: '',
    error: null,
    result: null,
  })

  const onCompleteRef = useRef<((result: IcLoraResult) => void) | undefined>()

  const submitIcLora = useCallback(async (params: IcLoraSubmitParams, onComplete?: (result: IcLoraResult) => void) => {
    onCompleteRef.current = onComplete

    setState({
      isGenerating: true,
      status: 'Generating',
      error: null,
      result: null,
    })

    const body: Record<string, unknown> = {
      conditioning_strength: params.conditioningStrength,
      lora_strength: params.loraStrength ?? undefined,
      prompt: params.prompt,
      frame_rate: params.frameRate ?? 24,
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
    const result = await ApiClient.generateIcLora(body as GenerateIcLoraBody)
    if (!result.ok) {
      logger.error(`IC-LoRA error: ${result.error.message}`)
      setState({
        isGenerating: false,
        status: '',
        error: result.error.message,
        result: null,
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
      })
      return
    }

    if (payload.status === 'complete') {
      const res: IcLoraResult = {
        videoPath: payload.video_path,
      }
      // Fire onComplete before local setState — runs ProjectContext mutations
      // even if GenSpace has unmounted (Bug A fix)
      onCompleteRef.current?.(res)
      onCompleteRef.current = undefined
      setState({
        isGenerating: false,
        status: 'Generation complete!',
        error: null,
        result: res,
      })
      return
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      status: '',
      error: null,
      result: null,
    })
  }, [])

  return {
    submitIcLora,
    resetIcLora: reset,
    isIcLoraGenerating: state.isGenerating,
    icLoraStatus: state.status,
    icLoraError: state.error,
    icLoraResult: state.result,
  }
}
