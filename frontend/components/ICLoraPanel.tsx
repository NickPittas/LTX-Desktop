import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import {
  Loader2, Film, Sparkles, Image as ImageIcon,
  RefreshCw, Download, AlertCircle, Trash2,
} from 'lucide-react'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { pathToFileUrl } from '../lib/file-url'
import { useHfAuth } from '../hooks/use-hf-auth'
import { useHfModelAccess } from '../hooks/use-hf-model-access'

const VIDEO_EXTS = new Set(['mp4', 'mov', 'avi', 'webm', 'mkv'])
const isVideoPath = (p: string) => VIDEO_EXTS.has(p.split('.').pop()?.toLowerCase() ?? '')

export type ICLoraConditioningType = 'canny' | 'depth' | null

interface ICLoraPanelProps {
  initialVideoPath?: string | null
  resetKey?: number
  fillHeight?: boolean
  isProcessing?: boolean
  processingStatus?: string
  conditioningType?: ICLoraConditioningType
  onConditioningTypeChange?: (type: ICLoraConditioningType) => void
  conditioningStrength?: number
  onConditioningStrengthChange?: (strength: number) => void
  outputVideoPath?: string | null
  onChange?: (data: {
    videoPath: string | null
    conditioningType: ICLoraConditioningType
    conditioningStrength: number
    adapterId: string | null
    maskPath: string | null
    images: { path: string; frame?: number; strength?: number }[]
    ready: boolean
    maskGrowPx: number
    laplacianBlendGrow: number
    finalMaskBlurPx: number
  }) => void
}

export type AdapterWorkflow = 'standard_video' | 'ingredients' | 'in_outpainting' | 'unavailable'

export interface AdapterEntry {
  value: string
  label: string
  workflow: AdapterWorkflow
  /** Only for unavailable adapters — explains why not wired yet. */
  reason?: string
}

// ponytail: flat registry, not from backend. If backend adds/excludes adapters,
// add a /api/ic-lora/adapters endpoint and fetch this dynamically.
export const IC_LORA_ADAPTERS: AdapterEntry[] = [
  // standard_video — source video only, no extra conditioning
  { value: 'water_simulation', label: 'Water Simulation', workflow: 'standard_video' },
  { value: 'decompression', label: 'Decompression', workflow: 'standard_video' },
  { value: 'deblur', label: 'Deblur', workflow: 'standard_video' },
  { value: 'colorization', label: 'Colorization', workflow: 'standard_video' },
  { value: 'day_to_night', label: 'Day to Night', workflow: 'standard_video' },
  { value: 'instant_shave', label: 'Instant Shave', workflow: 'standard_video' },
  { value: 'cross_eyed', label: 'Cross Eyed', workflow: 'standard_video' },
  // ingredients — needs one reference image
  { value: 'ingredients', label: 'Ingredients', workflow: 'ingredients' },
  // in_outpainting — inpaint supported; outpaint not yet
  { value: 'in_outpainting', label: 'In/Outpainting', workflow: 'in_outpainting' },
  // unavailable — visible but cannot be selected
  { value: 'motion_track_control', label: 'Motion Track Control', workflow: 'unavailable', reason: 'Motion track workflow not wired yet' },
  { value: 'hdr', label: 'HDR', workflow: 'unavailable', reason: 'HDR workflow not wired yet' },
  { value: 'lipdub', label: 'LipDub', workflow: 'unavailable', reason: 'LipDub requires audio workflow — not wired yet' },
]

const availableAdapters = IC_LORA_ADAPTERS.filter(a => a.workflow !== 'unavailable')
const unavailableAdapters = IC_LORA_ADAPTERS.filter(a => a.workflow === 'unavailable')

function getAdapterEntry(value: string | null): AdapterEntry | undefined {
  return IC_LORA_ADAPTERS.find(a => a.value === value)
}

export const CONDITIONING_TYPES: { value: ICLoraConditioningType; label: string; desc: string }[] = [
  { value: null, label: 'None', desc: 'No conditioning' },
  { value: 'canny', label: 'Canny Edges', desc: 'Edge detection' },
  { value: 'depth', label: 'Depth Map', desc: 'Estimated depth' },
]

type StartModelDownloadBody = NonNullable<ApiRequestBodyOf<'startModelDownload'>>
type ModelCheckpointID = NonNullable<StartModelDownloadBody['cp_ids']>[number]
type DownloadProgress = ApiSuccessOf<'getModelDownloadProgress'>


export function ICLoraPanel({
  initialVideoPath,
  resetKey,
  fillHeight = false,
  isProcessing = false,
  processingStatus = '',
  conditioningType: conditioningTypeProp,
  onConditioningTypeChange,
  conditioningStrength: conditioningStrengthProp,
  onConditioningStrengthChange,
  outputVideoPath: _outputVideoPath,
  onChange,
}: ICLoraPanelProps) {
  const inputVideoRef = useRef<HTMLVideoElement>(null)
  const [inputVideoPath, setInputVideoPath] = useState<string | null>(initialVideoPath || null)
  const inputVideoUrl = inputVideoPath ? pathToFileUrl(inputVideoPath) : null
  const [inputTime, setInputTime] = useState(0)

  const [internalCondType, setInternalCondType] = useState<ICLoraConditioningType>(null)
  const [internalCondStrength, setInternalCondStrength] = useState(1.0)
  const [internalAdapterId, setInternalAdapterId] = useState<string | null>(null)
  const conditioningType = conditioningTypeProp ?? internalCondType
  const conditioningStrength = conditioningStrengthProp ?? internalCondStrength
  const [conditioningPreview, setConditioningPreview] = useState<string | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)

  const [maskPath, setMaskPath] = useState<string | null>(null)
  const [maskGrowPx, setMaskGrowPx] = useState(30)
  const [laplacianBlendGrow, setLaplacianBlendGrow] = useState(12)
  const [finalMaskBlurPx, setFinalMaskBlurPx] = useState(6)
  const [ingredientPaths, setIngredientPaths] = useState<string[]>([])

  const showConditioning = internalCondType !== null
  const depthCpId = 'dpt-hybrid-midas' as ModelCheckpointID
  const needsDepthCp = showConditioning && conditioningType === 'depth'
  const [requiredIcLoraCpIds, setRequiredIcLoraCpIds] = useState<ModelCheckpointID[]>([])
  const [isCheckingIcLora, setIsCheckingIcLora] = useState(false)
  const [isDownloadingIcLora, setIsDownloadingIcLora] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [extractError, setExtractError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const allRequiredCpIds = useMemo(() => {
    return needsDepthCp && !requiredIcLoraCpIds.includes(depthCpId)
      ? [...requiredIcLoraCpIds, depthCpId]
      : requiredIcLoraCpIds
  }, [requiredIcLoraCpIds, needsDepthCp])
  const icLoraReady = allRequiredCpIds.length === 0
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(!icLoraReady)
  const { accessMap, allAuthorized } = useHfModelAccess(allRequiredCpIds, hfAuthStatus)

  useEffect(() => {
    if (resetKey === undefined) return
    setInputVideoPath(initialVideoPath || null)
    setInputTime(0)
    setInternalCondType(null)
    setInternalCondStrength(1.0)
    setInternalAdapterId(null)
    setMaskPath(null)
    setIngredientPaths([])
    setMaskGrowPx(30)
    setLaplacianBlendGrow(12)
    setFinalMaskBlurPx(6)
    onConditioningTypeChange?.(null)
    onConditioningStrengthChange?.(1.0)
    setConditioningPreview(null)
    setExtractError(null)
  }, [resetKey, initialVideoPath]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const selectedEntry = getAdapterEntry(internalAdapterId)
    const requiredSlotsReady = selectedEntry?.workflow === 'in_outpainting'
      ? !!maskPath
      : selectedEntry?.workflow === 'ingredients'
        ? ingredientPaths.length > 0
        : true
    const adapterReady = internalAdapterId !== null && selectedEntry?.workflow !== 'unavailable'
    const selectedWorkflow = selectedEntry?.workflow
    // ponytail: ingredients doesn't need driving video; conditioning is always null
    const needsVideo = selectedWorkflow !== 'ingredients'
    const ready = (needsVideo ? !!inputVideoPath : true) && icLoraReady && requiredSlotsReady && (conditioningType !== null || adapterReady)
    const images = selectedWorkflow === 'ingredients' ? ingredientPaths.map(p => ({ path: p })) : []
    onChange?.({
      videoPath: selectedWorkflow === 'ingredients' ? null : inputVideoPath,
      conditioningType: selectedWorkflow === 'ingredients' ? null : conditioningType,
      conditioningStrength,
      adapterId: internalAdapterId,
      maskPath: selectedEntry?.workflow === 'in_outpainting' ? maskPath : null,
      images,
      ready,
      maskGrowPx: selectedEntry?.workflow === 'in_outpainting' ? maskGrowPx : 30,
      laplacianBlendGrow: selectedEntry?.workflow === 'in_outpainting' ? laplacianBlendGrow : 12,
      finalMaskBlurPx: selectedEntry?.workflow === 'in_outpainting' ? finalMaskBlurPx : 6,
    })
  }, [inputVideoUrl, inputVideoPath, conditioningType, conditioningStrength, internalAdapterId, icLoraReady, maskPath, maskGrowPx, laplacianBlendGrow, finalMaskBlurPx, ingredientPaths, onChange])

  const checkIcLoraAvailability = useCallback(async () => {
    setIsCheckingIcLora(true)

    // 1. Generic IC-LoRA base CPs (depth, canny, pose control models)
    const loraResult = await ApiClient.getLtxIcLoraRecommendation()
    if (!loraResult.ok) {
      logger.warn(`Failed to fetch IC-LoRA model status: ${loraResult.error.message}`)
      setDownloadError(loraResult.error.message)
      setIsCheckingIcLora(false)
      return
    }

    const cpIds: ModelCheckpointID[] = [...loraResult.data.cps_to_download]

    // 2. Adapter-specific CPs from backend recommendation (no frontend adapter→CP map)
    if (internalAdapterId) {
      const adapterResult = await ApiClient.getAdapterRecommendation({ pipeline: internalAdapterId as any })
      if (adapterResult.ok) {
        for (const cpId of adapterResult.data.cps_to_download) {
          if (!cpIds.includes(cpId)) {
            cpIds.push(cpId)
          }
        }
      }
    }

    setRequiredIcLoraCpIds(cpIds)
    const isReady = cpIds.length === 0

    if (isReady) {
      setIsDownloadingIcLora(false)
      setDownloadProgress(null)
      setDownloadError(null)
      setDownloadSessionId(null)
    }
    setIsCheckingIcLora(false)
  }, [internalAdapterId])

  useEffect(() => {
    void checkIcLoraAvailability()
  }, [checkIcLoraAvailability])

  useEffect(() => {
    if (icLoraReady || !isDownloadingIcLora || !downloadSessionId) return

    const pollProgress = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!result.ok) {
        logger.warn(`Failed polling IC-LoRA download progress: ${result.error.message}`)
        return
      }

      const progressPayload = result.data
      setDownloadProgress(progressPayload)

      if (progressPayload.status === 'error') {
        setIsDownloadingIcLora(false)
        setDownloadError(progressPayload.error || 'Model download failed')
        return
      }

      if (progressPayload.status === 'complete') {
        setIsDownloadingIcLora(false)
        await checkIcLoraAvailability()
      }
    }

    void pollProgress()
    const interval = setInterval(() => { void pollProgress() }, 1000)
    return () => clearInterval(interval)
  }, [icLoraReady, isDownloadingIcLora, downloadSessionId, checkIcLoraAvailability])

  const handleDownloadIcLora = useCallback(async () => {
    if (isDownloadingIcLora) return
    setDownloadError(null)

    const result = await ApiClient.startModelDownload({
      type: 'download',
      cp_ids: [...allRequiredCpIds],
    })
    if (!result.ok) {
      logger.warn(`Failed to start IC-LoRA download: ${result.error.message}`)
      setDownloadError(result.error.message)
      return
    }

    const startedPayload = result.data
    if (startedPayload.status === 'started') {
      setDownloadSessionId(startedPayload.sessionId)
      setIsDownloadingIcLora(true)
      return
    }

    setDownloadError('Unexpected response while starting IC-LoRA download')
  }, [isDownloadingIcLora, allRequiredCpIds])

  const isExtractingRef = useRef(false)
  const extractConditioning = useCallback(async () => {
    if (!inputVideoPath || isExtractingRef.current || !icLoraReady || !showConditioning) return
    isExtractingRef.current = true
    setIsExtracting(true)
    setExtractError(null)
    if (conditioningType === null) return
    const result = await ApiClient.extractIcLoraConditioning({
      video_path: inputVideoPath,
      conditioning_type: conditioningType,
      frame_time: inputTime,
    })
    if (!result.ok) {
      logger.warn(`Failed to extract conditioning: ${result.error.message}`)
      setExtractError(result.error.message)
      isExtractingRef.current = false
      setIsExtracting(false)
      return
    }

    setConditioningPreview(result.data.conditioning)
    isExtractingRef.current = false
    setIsExtracting(false)
  }, [inputVideoPath, conditioningType, inputTime, icLoraReady])

  const extractTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (!inputVideoPath || !icLoraReady || !showConditioning) return
    if (extractTimerRef.current) clearTimeout(extractTimerRef.current)
    extractTimerRef.current = setTimeout(() => {
      void extractConditioning()
    }, 300)
    return () => {
      if (extractTimerRef.current) clearTimeout(extractTimerRef.current)
    }
  }, [inputTime, conditioningType, inputVideoPath, icLoraReady, extractConditioning, showConditioning])

  useEffect(() => {
    const video = inputVideoRef.current
    if (!video) return
    const onTime = () => setInputTime(video.currentTime)
    const onSeeked = () => setInputTime(video.currentTime)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('seeked', onSeeked)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('seeked', onSeeked)
    }
  }, [inputVideoUrl, icLoraReady, isCheckingIcLora])

  const handleBrowse = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select Driving Video',
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'webm', 'mkv'] }],
    })
    if (paths && paths.length > 0) {
      const filePath = paths[0]
      setInputVideoPath(filePath)
      
      setConditioningPreview(null)
      setExtractError(null)
    }
  }, [])

  const handlePickImage = useCallback(async (title: string) => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title,
      filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp'] }],
    })
    return paths?.[0] ?? null
  }, [])

  const handlePickMaskFile = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select Mask Video or Image',
      filters: [
        { name: 'Video', extensions: ['mp4', 'mov', 'avi', 'webm', 'mkv'] },
        { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp'] },
      ],
    })
    return paths?.[0] ?? null
  }, [])

  const handleClear = useCallback(() => {
    setInputVideoPath(null)
    setMaskPath(null)
    setIngredientPaths([])
    setInputTime(0)
    setConditioningPreview(null)
    setExtractError(null)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      try {
        const asset = JSON.parse(assetData) as { type?: string; path?: string }
        if (asset.type === 'video' && asset.path) {
          setInputVideoPath(asset.path)
          setConditioningPreview(null)
          setExtractError(null)
          return
        }
      } catch {
        // fall through
      }
    }

    const file = e.dataTransfer.files?.[0]
    if (file) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        setInputVideoPath(filePath)
        
        setConditioningPreview(null)
        setExtractError(null)
      }
    }
  }, [])

  const showDownloadGate = isCheckingIcLora || !icLoraReady
  const runningDownloadProgress =
    downloadProgress?.status === 'downloading' ? downloadProgress : null
  const gateItemIds = [...new Set([...(allRequiredCpIds ?? []), ...(runningDownloadProgress?.all_files ?? [])])]
  const gateItems = gateItemIds.map((cpId) => {
    const downloaded = !allRequiredCpIds.includes(cpId)
    const isCompleted = runningDownloadProgress?.completed_files?.includes(cpId) ?? false
    const isCurrentDownload = isDownloadingIcLora && runningDownloadProgress?.current_downloading_file === cpId
    const progress = downloaded ? 100 : (isCompleted ? 100 : (isCurrentDownload ? (runningDownloadProgress?.current_file_progress ?? 0) : 0))
    const status = downloaded ? 'Ready' : (isCompleted ? 'Complete' : (isCurrentDownload ? 'Downloading' : 'Missing'))
    return { id: cpId, label: cpId, downloaded, progress, status }
  })

  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden flex flex-col ${fillHeight ? 'h-full min-h-0' : ''}`}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-amber-400" />
          <span className="text-sm font-semibold text-white">IC-LoRA / Style Transfer</span>
        </div>
        {inputVideoUrl && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleClear}
              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
              title="Clear video"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={handleBrowse}
              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
              title="Replace video"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>

      {showDownloadGate ? (
        <div className="flex-1 flex items-center justify-center p-6 min-h-0 overflow-y-auto">
          <div className="w-full max-w-xl rounded-xl border border-zinc-700 bg-zinc-800/60 p-6">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-lg bg-blue-600/20 flex items-center justify-center mt-0.5">
                <Download className="h-4 w-4 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold text-white">Download Required: IC-LoRA Resources</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Editing is locked until all IC-LoRA preprocessing models are available locally.
                </p>
              </div>
            </div>

            <div className="mt-5 space-y-3">
              {isCheckingIcLora ? (
                <div className="flex items-center gap-2 text-xs text-zinc-300">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                  Checking model availability...
                </div>
              ) : (
                <>
                  <div className="space-y-2">
                    {gateItems.map(item => (
                      <div key={item.id} className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-2">
                        <div className="flex items-center justify-between text-[11px] mb-1.5">
                          <span className="text-zinc-300">{item.label}</span>
                          <span className={item.downloaded ? 'text-blue-400' : 'text-zinc-500'}>
                            {item.status}
                          </span>
                        </div>
                        <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full transition-all duration-300 bg-blue-500"
                            style={{ width: `${item.progress}%` }}
                          />
                        </div>
                        <div className="mt-1 text-[10px] text-zinc-500">{item.progress}%</div>
                      </div>
                    ))}
                  </div>
                  {downloadError && (
                    <div className="text-[11px] text-red-400">{downloadError}</div>
                  )}
                  {hfAuthStatus === 'authenticated' && !allAuthorized && Object.keys(accessMap).length > 0 && (
                    <div className="space-y-1.5 pt-1 pb-1">
                      <div className="text-[11px] text-amber-400">Accept license for these models:</div>
                      {Object.entries(accessMap)
                        .filter(([, status]) => status === 'not_authorized')
                        .map(([repoId]) => (
                          <div key={repoId} className="flex items-center justify-between bg-zinc-900 rounded px-2 py-1.5">
                            <span className="text-[10px] text-zinc-400 font-mono">{repoId}</span>
                            <button
                              onClick={() => window.electronAPI.openHuggingFaceRepo({ repoId })}
                              className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium"
                            >
                              Request access
                            </button>
                          </div>
                        ))}
                    </div>
                  )}
                  <div className="flex items-center gap-2 pt-1">
                    {hfAuthStatus !== 'authenticated' ? (
                      <button
                        onClick={startHuggingFaceLogin}
                        disabled={hfAuthPolling}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {hfAuthPolling ? (
                          <>
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Waiting for sign in...
                          </>
                        ) : (
                          'Sign in with HuggingFace'
                        )}
                      </button>
                    ) : (
                      <button
                        onClick={handleDownloadIcLora}
                        disabled={isDownloadingIcLora || !allAuthorized}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isDownloadingIcLora ? (
                          <>
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Downloading...
                          </>
                        ) : (
                          <>
                            <Download className="h-3 w-3" />
                            {downloadError ? 'Retry Download' : 'Download Models'}
                          </>
                        )}
                      </button>
                    )}
                    <button
                      onClick={() => { void checkIcLoraAvailability() }}
                      disabled={isCheckingIcLora}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-600 text-zinc-300 hover:text-white hover:border-zinc-500 text-xs transition-colors disabled:opacity-50"
                    >
                      <RefreshCw className={`h-3 w-3 ${isCheckingIcLora ? 'animate-spin' : ''}`} />
                      Refresh
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex min-h-0 overflow-hidden">
          <div className="flex-1 flex flex-col border-r border-zinc-800 min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-2">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider shrink-0">Input</span>
              {getAdapterEntry(internalAdapterId)?.workflow === 'ingredients' ? (
                <span className="text-[10px] text-amber-500 truncate min-w-0">Ingredients — prompt + reference sheet</span>
              ) : inputVideoPath ? (
                <span className="text-[10px] text-zinc-500 truncate min-w-0">
                  {inputVideoPath.split(/[\\/]/).pop()}
                </span>
              ) : null}
            </div>
            {getAdapterEntry(internalAdapterId)?.workflow === 'ingredients' ? (
              <div className="flex-1 flex items-center justify-center bg-zinc-900/50 m-3 rounded-lg border border-dashed border-zinc-700">
                <div className="text-center p-6">
                  <div className="w-12 h-12 rounded-full bg-amber-500/10 flex items-center justify-center mx-auto mb-3">
                    <Sparkles className="h-6 w-6 text-amber-500" />
                  </div>
                  <p className="text-zinc-300 text-sm font-medium">Ingredients</p>
                  <p className="text-zinc-500 text-xs mt-1 max-w-xs">
                    Uses prompt + reference sheet image. No driving video needed.
                  </p>

                </div>
              </div>
            ) : (
              <div
                className={`flex-1 min-h-0 bg-black flex items-center justify-center relative ${!inputVideoUrl ? 'border-2 border-dashed border-zinc-700 m-3 rounded-lg' : ''} ${isDragOver ? 'border-blue-500 bg-blue-500/10' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
              >
                {inputVideoUrl ? (
                  <video
                    ref={inputVideoRef}
                    src={inputVideoUrl}
                    className="w-full h-full object-contain"
                    controls
                    onError={(e) => console.error('[ICLoraPanel] Input video failed to load:', inputVideoUrl, (e.target as HTMLVideoElement)?.error)}
                  />
                ) : (
                  <div className="text-center p-4">
                    <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-2">
                      <Film className="h-6 w-6 text-zinc-600" />
                    </div>
                    <p className="text-zinc-400 text-xs">Drop or import a driving video</p>
                    <button
                      onClick={handleBrowse}
                      className="mt-2 px-3 py-1.5 text-[10px] text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-600/10 transition-colors"
                    >
                      Import Video
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {showConditioning ? (
            <div className="flex-1 flex flex-col min-w-0">
              <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-2">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Conditioning</span>
                <button
                  onClick={() => { void extractConditioning() }}
                  disabled={!inputVideoPath || isExtracting}
                  className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={`h-3 w-3 ${isExtracting ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
                {isExtracting && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                    <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                  </div>
                )}
                {conditioningPreview ? (
                  <img src={conditioningPreview} alt="Conditioning preview" className="w-full h-full object-contain" />
                ) : (
                  <div className="text-center p-4">
                    <p className="text-zinc-600 text-xs">
                      {inputVideoUrl ? 'Scrub the input video to see conditioning preview' : 'Import a video to preview conditioning'}
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : getAdapterEntry(internalAdapterId)?.workflow === 'in_outpainting' ? (
            <div className="flex-1 flex flex-col min-w-0">
              <div className="px-3 py-2 border-b border-zinc-800 flex flex-col gap-1">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Mask video/image</span>
                <p className="text-[10px] text-zinc-500 leading-relaxed">
                  White = inpaint, black = keep.&nbsp; Inpaint currently supported; outpaint not yet.
                </p>
              </div>
              <div className="flex-none px-3 py-2 border-b border-zinc-800 space-y-2">
                <label className="flex items-center gap-2">
                  <span className="text-[10px] text-zinc-400 shrink-0">Mask grow:</span>
                  <span className="text-[9px] text-zinc-600 shrink-0">mask dilation</span>
                  <input
                    type="range"
                    min={0}
                    max={128}
                    value={maskGrowPx}
                    onChange={(e) => setMaskGrowPx(Number(e.target.value))}
                    className="w-full h-1.5 accent-blue-500"
                  />
                  <span className="text-[10px] text-zinc-400 w-6 text-right tabular-nums">{maskGrowPx}</span>
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-[10px] text-zinc-400 shrink-0">Blend grow:</span>
                  <span className="text-[9px] text-zinc-600 shrink-0">Laplacian mask</span>
                  <input
                    type="range"
                    min={0}
                    max={64}
                    value={laplacianBlendGrow}
                    onChange={(e) => setLaplacianBlendGrow(Number(e.target.value))}
                    className="w-full h-1.5 accent-blue-500"
                  />
                  <span className="text-[10px] text-zinc-400 w-6 text-right tabular-nums">{laplacianBlendGrow}</span>
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-[10px] text-zinc-400 shrink-0">Final blur:</span>
                  <span className="text-[9px] text-zinc-600 shrink-0">edge feather</span>
                  <input
                    type="range"
                    min={0}
                    max={64}
                    value={finalMaskBlurPx}
                    onChange={(e) => setFinalMaskBlurPx(Number(e.target.value))}
                    className="w-full h-1.5 accent-blue-500"
                  />
                  <span className="text-[10px] text-zinc-400 w-6 text-right tabular-nums">{finalMaskBlurPx}</span>
                </label>
              </div>
              <div className="flex-1 flex items-center justify-center min-h-0 p-4">
                {maskPath ? (
                  <div className="text-center">
                    {isVideoPath(maskPath) ? (
                      <video
                        src={pathToFileUrl(maskPath)}
                        className="max-w-full max-h-40 object-contain mx-auto mb-2"
                        controls
                        muted
                        playsInline
                        onError={(e) => console.error('[ICLoraPanel] Mask video failed to load:', maskPath, (e.target as HTMLVideoElement)?.error)}
                      />
                    ) : (
                      <img
                        src={pathToFileUrl(maskPath)}
                        alt="Mask"
                        className="max-w-full max-h-40 object-contain mx-auto mb-2"
                        onError={() => console.error('[ICLoraPanel] Mask image failed to load:', maskPath)}
                      />
                    )}
                    <p className="text-zinc-400 text-[10px] truncate max-w-[150px] mx-auto">{maskPath.split(/[\\/]/).pop()}</p>
                    <button onClick={() => setMaskPath(null)} className="text-[10px] text-red-400 hover:text-red-300 mt-1">Remove</button>
                  </div>
                ) : (
                  <button
                    onClick={() => { void handlePickMaskFile().then(setMaskPath) }}
                    className="flex flex-col items-center gap-1 px-3 py-2 text-[10px] text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-600/10 transition-colors"
                  >
                    <ImageIcon className="h-5 w-5" />
                    Select mask video or image
                  </button>
                )}
              </div>
            </div>
          ) : getAdapterEntry(internalAdapterId)?.workflow === 'ingredients' ? (
            <div className="flex-1 flex flex-col min-w-0">
              <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Ingredient Images</span>
                <button
                  onClick={() => { void handlePickImage('Add Ingredient Image').then(p => p && setIngredientPaths(prev => [...prev, p])) }}
                  disabled={isProcessing}
                  className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-blue-400 hover:text-blue-300 hover:bg-blue-600/10 transition-colors disabled:opacity-50"
                >
                  + Add
                </button>
              </div>
              <div className="flex-1 overflow-y-auto min-h-0 p-3 space-y-2">
                <p className="text-[10px] text-zinc-500 leading-relaxed">
                  Official workflow expects a single composite reference sheet; multiple assets are passed as separate references.
                </p>
                {ingredientPaths.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-6 text-center">
                    <ImageIcon className="h-8 w-8 text-zinc-700 mb-2" />
                    <p className="text-zinc-500 text-[10px]">No ingredient images selected</p>
                    <p className="text-zinc-600 text-[10px] mt-1">Add at least one reference image</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    {ingredientPaths.map((path, idx) => (
                      <div key={idx} className="relative group rounded-lg border border-zinc-700 bg-zinc-900 overflow-hidden">
                        <div className="aspect-square bg-black flex items-center justify-center">
                          <img
                            src={pathToFileUrl(path)}
                            alt={`Ingredient ${idx + 1}`}
                            className="w-full h-full object-contain"
                          />
                        </div>
                        <div className="px-2 py-1.5">
                          <p className="text-[10px] text-zinc-400 truncate">{path.split(/[\\/]/).pop()}</p>
                        </div>
                        <button
                          onClick={() => setIngredientPaths(prev => prev.filter((_, i) => i !== idx))}
                          className="absolute top-1 right-1 p-1 rounded bg-black/60 text-red-400 opacity-0 group-hover:opacity-100 hover:text-red-300 transition-opacity"
                          title="Remove ingredient"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col min-w-0">
              <div className="flex-1 flex items-center justify-center p-4">
                <p className="text-zinc-600 text-xs text-center">No conditioning needed for this adapter</p>
              </div>
            </div>
          )}

          {/* Output column */}
          <div className="flex-1 flex flex-col border-l border-zinc-800 min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center gap-2">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Output</span>
              <div className="flex-1" />
              <select
                value={internalAdapterId ?? ''}
                onChange={(e) => {
                  const val = e.target.value
                  if (val && getAdapterEntry(val)?.workflow === 'unavailable') return // guard against disabled selection
                  setInternalAdapterId(val || null)
                  setMaskPath(null)
                  setIngredientPaths([])
                }}
                className="bg-zinc-800 text-[10px] text-zinc-300 border border-zinc-700 rounded px-1.5 py-1 max-w-[140px] cursor-pointer focus:outline-none focus:border-zinc-500"
              >
                <option value="">Default</option>
                {availableAdapters.map((a) => (
                  <option key={a.value} value={a.value}>{a.label}</option>
                ))}
                {unavailableAdapters.length > 0 && (
                  <option disabled value="">─── Unavailable ───</option>
                )}
                {unavailableAdapters.map((a) => (
                  <option key={a.value} value={a.value} disabled>{a.label} ({a.reason})</option>
                ))}
              </select>
            </div>
            <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
              {_outputVideoPath ? (
                <video
                  src={pathToFileUrl(_outputVideoPath)}
                  className="w-full h-full object-contain"
                  controls
                  onError={(e) => console.error('[ICLoraPanel] Output video failed to load:', _outputVideoPath, (e.target as HTMLVideoElement)?.error)}
                />
              ) : isProcessing ? (
                <div className="text-center p-4">
                  <Loader2 className="h-6 w-6 text-blue-400 animate-spin mx-auto mb-2" />
                  <p className="text-zinc-400 text-xs">{processingStatus || 'Generating...'}</p>
                </div>
              ) : (
                <div className="text-center p-4">
                  <p className="text-zinc-600 text-xs">Output video will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {extractError && (
        <div className="px-4 py-3 border-t border-zinc-800 flex-shrink-0">
          <div className="flex items-center gap-2 text-xs text-red-400">
            <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
            <span>{extractError}</span>
          </div>
        </div>
      )}
    </div>
  )
}
