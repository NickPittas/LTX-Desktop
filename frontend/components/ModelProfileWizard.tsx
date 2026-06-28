import { AlertTriangle, ArrowLeft, ArrowRight, Check, Loader2 } from 'lucide-react'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type { ModelComponentPaths, ModelProfilePayload, ModelProfileValidationResponse } from '../types/model-profile'
import { Button } from './ui/button'
import { ModelComponentPicker, MODEL_FILE_FILTERS } from './ModelComponentPicker'
import type { ActivationError } from '../hooks/use-model-profiles'

// ── types ──────────────────────────────────────────────────────────────

export interface ModelProfileWizardProps {
  isOpen: boolean
  onClose: () => void
  onCreated?: (profileId: string) => void
  generationRunning?: boolean
}

type Capability = NonNullable<ModelProfilePayload['capabilities']>[number]

// Keep in sync with backend CURRENT_MODEL_PROFILE_SCHEMA_VERSION (api_types.py).
const CURRENT_MODEL_PROFILE_SCHEMA_VERSION = 1

function parseActivationError(error: { code?: string; message: string }): ActivationError {
  const code = error.code ?? 'UNKNOWN'
  const knownCodes: string[] = [
    'MODEL_PROFILE_ACTIVATION_GENERATION_RUNNING',
    'MODEL_PROFILE_REQUIRED_ARTIFACTS_MISSING',
    'MODEL_PROFILE_CHANGED_DURING_ACTIVATION',
  ]
  if (knownCodes.includes(code)) {
    return { code: code as ActivationError['code'], message: error.message }
  }
  return { code: 'UNKNOWN', message: error.message }
}

function formatActivationError(error: ActivationError): string {
  switch (error.code) {
    case 'MODEL_PROFILE_ACTIVATION_GENERATION_RUNNING':
      return 'A generation is currently running. Wait for it to finish, then activate this profile.'
    case 'MODEL_PROFILE_REQUIRED_ARTIFACTS_MISSING':
      return 'Some required model files are missing. Use the Library tab to check what is installed and download missing items.'
    case 'MODEL_PROFILE_CHANGED_DURING_ACTIVATION':
      return 'The profile changed while activating. Rescan and try again.'
    default:
      return error.message
  }
}

// ── component field definitions ────────────────────────────────────────

interface ComponentFieldDef {
  key: keyof ModelComponentPaths
  label: string
  /** If undefined, source-dependent. If set, overrides. */
  pickDirectory?: boolean
}

const COMPONENT_FIELDS: ComponentFieldDef[] = [
  { key: 'transformer', label: 'Transformer' },
  { key: 'text_encoder_root', label: 'Text Encoder' },
  { key: 'video_vae', label: 'Video VAE' },
  { key: 'upsampler', label: 'Upsampler' },
  { key: 'vocoder', label: 'Vocoder' },
  { key: 'audio_vae', label: 'Audio VAE' },
  { key: 'person_detector', label: 'Person Detector' },
  { key: 'depth_processor', label: 'Depth Processor' },
  { key: 'pose_processor', label: 'Pose Processor' },
  { key: 'embeddings_connector', label: 'Embeddings Connector' },
  { key: 'text_projection', label: 'Text Projection' },
  { key: 'ic_lora_union', label: 'IC-LoRA Union' },
  { key: 'ic_lora_lipdub', label: 'IC-LoRA Lipdub' },
  { key: 'ic_lora_hdr', label: 'IC-LoRA HDR' },
  { key: 'ic_lora_hdr_scene_embeddings', label: 'IC-LoRA HDR Scene Embeddings' },
  { key: 'ic_lora_in_outpainting', label: 'IC-LoRA In-Outpainting' },
  { key: 'ic_lora_ingredients', label: 'IC-LoRA Ingredients' },
  { key: 'ic_lora_motion_track', label: 'IC-LoRA Motion Track' },
  { key: 'transformer_quantization', label: 'Transformer Quantization' },
]

// ── prefill candidates ──────────────────────────────────────────
// ponytail: known QuantStack/Kijai layout under a single root.
const PREFILL_CANDIDATES: Record<string, string> = {
  transformer: 'gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf',
  text_encoder_root: 'text_encoders/unsloth/gemma-3-12b-it-qat-GGUF/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf',
  text_projection: 'text_encoders/ltx-2.3_text_projection_bf16.safetensors',
  video_vae: 'vae/LTX23_video_vae_bf16.safetensors',
  audio_vae: 'vae/LTX23_audio_vae_bf16.safetensors',
  upsampler: 'ltx-2.3-spatial-upscaler-x2-1.0.safetensors',
  ic_lora_union: 'adapters/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors',
  ic_lora_motion_track: 'adapters/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors',
  ic_lora_ingredients: 'adapters/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
  ic_lora_hdr: 'adapters/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors',
  ic_lora_hdr_scene_embeddings: 'adapters/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors',
  ic_lora_lipdub: 'adapters/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors',
  ic_lora_in_outpainting: 'adapters/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors',
}

// ── official IC-LoRA adapter names (from OFFICIAL_LTX23_ADAPTERS) ──
// ponytail: all 13 IC-LoRA adapters; excludes distilled_lora_384, distilled_lora_384_1_1, hdr_scene_embeddings (non-ic_lora kind).
const OFFICIAL_ADAPTER_FILENAMES: Record<string, string> = {
  union_control: 'adapters/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors',
  motion_track_control: 'adapters/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors',
  ingredients: 'adapters/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
  water_simulation: 'adapters/ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors',
  decompression: 'adapters/ltx-2.3-22b-ic-lora-decompression-0.9.safetensors',
  deblur: 'adapters/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors',
  colorization: 'adapters/ltx-2.3-22b-ic-lora-colorization-0.9.safetensors',
  day_to_night: 'adapters/ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors',
  in_outpainting: 'adapters/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors',
  instant_shave: 'adapters/ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors',
  cross_eyed: 'adapters/ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors',
  hdr: 'adapters/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors',
  lipdub: 'adapters/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors',
}

const OFFICIAL_ADAPTER_LABELS: Record<string, string> = {
  union_control: 'Union Control',
  motion_track_control: 'Motion Track',
  ingredients: 'Ingredients',
  water_simulation: 'Water Simulation',
  decompression: 'Decompression',
  deblur: 'Deblur',
  colorization: 'Colorization',
  day_to_night: 'Day→Night',
  in_outpainting: 'In/Outpainting',
  instant_shave: 'Instant Shave',
  cross_eyed: 'Cross Eyed',
  hdr: 'HDR',
  lipdub: 'LipDub',
}

const OFFICIAL_ADAPTER_IDS = Object.keys(OFFICIAL_ADAPTER_FILENAMES)

// ponytail: best-effort capability guess. Backend validation is source of truth.
function deriveCapabilities(
  components: Partial<ModelComponentPaths>,
  transformerFormat: string,
  source: string,
): Capability[] {
  const caps: Capability[] = []

  if (components.transformer) {
    caps.push('t2v')
    if (components.video_vae) caps.push('i2v')
    if (components.upsampler) caps.push('retake')
  }
  if (
    Object.entries(components).some(([k, v]) => k.startsWith('ic_lora_') && v) ||
    Object.keys(components.official_adapters ?? {}).length > 0
  ) {
    caps.push('ic_lora')
  }
  if (transformerFormat === 'gguf') caps.push('gguf')
  if (source !== 'official') caps.push('local_text')

  return caps
}

function fieldPickDirectory(key: keyof ModelComponentPaths, transformerFormat: string, textEncoderFormat: string): boolean {
  if (key === 'transformer') return transformerFormat === 'split_safetensors'
  if (key === 'text_encoder_root') return textEncoderFormat === 'hf_folder'
  return false
}

function fieldFilters(key: keyof ModelComponentPaths, transformerFormat: string, textEncoderFormat: string) {
  if (key === 'transformer') {
    return transformerFormat === 'gguf' ? MODEL_FILE_FILTERS.gguf : MODEL_FILE_FILTERS.safetensors
  }
  if (key === 'text_encoder_root') {
    if (textEncoderFormat === 'gguf') return MODEL_FILE_FILTERS.gguf
    if (textEncoderFormat === 'safetensors') return MODEL_FILE_FILTERS.safetensors
    return MODEL_FILE_FILTERS.all
  }
  return MODEL_FILE_FILTERS.all
}

function visibleFields(source: ModelProfilePayload['source']): (keyof ModelComponentPaths)[] {
  // ponytail: hardcoded field sets. Expand when new sources added.
  const base: (keyof ModelComponentPaths)[] = ['transformer', 'text_encoder_root', 'video_vae', 'upsampler']
  if (source === 'official') return base
  return [
    ...base,
    'vocoder',
    'audio_vae',
    'person_detector',
    'depth_processor',
    'pose_processor',
    'embeddings_connector',
    'text_projection',
    'ic_lora_union',
    'ic_lora_lipdub',
    'ic_lora_hdr',
    'ic_lora_hdr_scene_embeddings',
    'ic_lora_in_outpainting',
    'ic_lora_ingredients',
    'ic_lora_motion_track',
    'transformer_quantization',
  ]
}

// ── wizard ─────────────────────────────────────────────────────────────

export function ModelProfileWizard({ isOpen, onClose, onCreated, generationRunning = false }: ModelProfileWizardProps) {
  const [step, setStep] = useState(0)
  const [name, setName] = useState('')
  const [family, setFamily] = useState<ModelProfilePayload['family']>('ltx-2.3')
  const [source, setSource] = useState<ModelProfilePayload['source']>('official')
  const [notes, setNotes] = useState('')
  const [transformerFormat, setTransformerFormat] = useState<string>('official_safetensors')
  const [textEncoderFormat, setTextEncoderFormat] = useState<string>('api')
  const [components, setComponents] = useState<Partial<ModelComponentPaths>>({})
  const [isCreating, setIsCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const [createdProfileId, setCreatedProfileId] = useState<string | null>(null)
  const [prefillStatus, setPrefillStatus] = useState<string | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [validationResult, setValidationResult] = useState<ModelProfileValidationResponse | null>(null)
  const [isActivating, setIsActivating] = useState(false)
  const [activateError, setActivateError] = useState<string | null>(null)
  const overlayRef = useRef<HTMLDivElement>(null)

  // Reset on open/close
  useEffect(() => {
    if (!isOpen) return
    setStep(0)
    setName('')
    setFamily('ltx-2.3')
    setSource('official')
    setNotes('')
    setTransformerFormat('official_safetensors')
    setTextEncoderFormat('api')
    setComponents({})
    setIsCreating(false)
    setCreateError(null)
    setCreatedProfileId(null)
    setIsValidating(false)
    setValidationResult(null)
    setIsActivating(false)
    setActivateError(null)
    setPrefillStatus(null)
  }, [isOpen])

  // Derive format defaults from source
  useEffect(() => {
    if (source === 'official') {
      setTransformerFormat('official_safetensors')
      setTextEncoderFormat('api')
    } else {
      if (transformerFormat === 'official_safetensors') setTransformerFormat('split_safetensors')
      if (textEncoderFormat === 'api') setTextEncoderFormat('hf_folder')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  const visibleKeys = useMemo(() => visibleFields(source), [source])
  const capabilities = useMemo(
    () => deriveCapabilities(components, transformerFormat, source),
    [components, transformerFormat, source],
  )

  const updateComponent = useCallback((key: keyof ModelComponentPaths, value: string) => {
    setComponents(prev => ({ ...prev, [key]: value || null }))
  }, [])

  const handlePrefill = useCallback(async () => {
    const dir = await window.electronAPI.showOpenDirectoryDialog({ title: 'Select models folder' })
    if (!dir) return

    setPrefillStatus('Scanning...')

    // Check standard component files
    const stdPairs = Object.entries(PREFILL_CANDIDATES).map(([k, rel]) => [k, `${dir}/${rel}`] as const)
    let exists: Record<string, boolean>
    try {
      const allPaths: string[] = stdPairs.map(([, p]) => p)
      OFFICIAL_ADAPTER_IDS.forEach(id => allPaths.push(`${dir}/${OFFICIAL_ADAPTER_FILENAMES[id]}`))
      exists = await window.electronAPI.checkFilesExist({
        filePaths: allPaths,
      })
    } catch {
      setPrefillStatus('Failed to check files')
      return
    }

    // Collect standard component paths
    const updates: Record<string, string> = {}
    for (const [key, absPath] of stdPairs) {
      if (exists[absPath]) updates[key] = absPath
    }

    // Collect official adapter paths
    const adapterDict: Record<string, string> = {}
    for (const id of OFFICIAL_ADAPTER_IDS) {
      const absPath = `${dir}/${OFFICIAL_ADAPTER_FILENAMES[id]}`
      if (exists[absPath]) adapterDict[id] = absPath
    }
    if (Object.keys(adapterDict).length > 0) {
      updates['official_adapters'] = adapterDict as unknown as string
    }

    const n = Object.keys(updates).length
    if (n === 0) {
      setPrefillStatus('No known files found')
      return
    }

    setComponents(prev => ({ ...prev, ...(updates as Partial<ModelComponentPaths>) }))

    // Set derived formats for GGUF files
    if (updates['transformer']) setTransformerFormat('gguf')
    if (updates['transformer']) setSource('quantstack')
    if (updates['text_encoder_root']) setTextEncoderFormat('gguf')

    const stdCount = Object.keys(updates).filter(k => k !== 'official_adapters').length
    const adapterCount = Object.keys(adapterDict).length
    setPrefillStatus(`Prefilled ${stdCount} paths + ${adapterCount} adapters from ${dir}`)
  }, [])

  // Step navigation
  const canGoNext = step === 0
    ? name.trim().length > 0
    : step === 1
      ? true // components are optional
      : false // review is last step

  const handleNext = () => {
    if (step < 2) setStep(s => s + 1)
  }
  const handlePrev = () => {
    if (step > 0) {
      setStep(s => s - 1)
      setCreateError(null)
    }
  }

  // Create + validate + activate flow
  const handleCreate = useCallback(async () => {
    setIsCreating(true)
    setCreateError(null)

    // Build full ModelComponentPaths — only include non-empty values
    const componentPaths: ModelComponentPaths = {
      transformer_format: transformerFormat as ModelComponentPaths['transformer_format'],
      text_encoder_format: textEncoderFormat as ModelComponentPaths['text_encoder_format'],
      ...Object.fromEntries(
        Object.entries(components).filter(([k, v]) => {
          if (k === 'official_adapters') {
            const d = v as Record<string, string> | undefined
            return d !== undefined && Object.keys(d).length > 0
          }
          return v
        }),
      ),
    } as ModelComponentPaths

    const payload: ModelProfilePayload = {
      id: '', // ponytail: backend assigns id on create
      name: name.trim(),
      family,
      source,
      notes: notes.trim(),
      created_at: '',
      updated_at: '',
      capabilities,
      components: componentPaths,
      schema_version: CURRENT_MODEL_PROFILE_SCHEMA_VERSION,
      created_by: 'wizard',
      validation_status: 'candidate',
      last_scanned_at: null,
      problems: [],
    }

    const result = await ApiClient.createModelProfile(payload)
    if (!result.ok) {
      setCreateError(result.error.message)
      setIsCreating(false)
      return
    }
    setCreatedProfileId(result.data.id)
    setCreateError(null)
    setIsCreating(false)
  }, [name, family, source, notes, components, capabilities, transformerFormat, textEncoderFormat])

  const handleValidate = useCallback(async () => {
    if (!createdProfileId) return
    setIsValidating(true)
    const result = await ApiClient.validateModelProfile(createdProfileId)
    if (result.ok) {
      setValidationResult(result.data)
    } else {
      setValidationResult({ valid: false, issues: [{ field: '', issue: result.error.message }] })
    }
    setIsValidating(false)
  }, [createdProfileId])

  const handleActivate = useCallback(async () => {
    if (!createdProfileId) return
    setIsActivating(true)
    setActivateError(null)
    const result = await ApiClient.activateModelProfile(createdProfileId)
    if (result.ok) {
      onCreated?.(createdProfileId)
      onClose()
    } else {
      setActivateError(formatActivationError(parseActivationError(result.error)))
    }
    setIsActivating(false)
  }, [createdProfileId, onCreated, onClose])

  // Close on overlay click
  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === overlayRef.current && !isCreating && !isActivating) {
      onClose()
    }
  }

  if (!isOpen) return null

  return (
    <div
      ref={overlayRef}
      onClick={handleOverlayClick}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      role="dialog"
      aria-modal="true"
      aria-label="Create Model Profile Wizard"
    >
      <div className="bg-card border border-border rounded-lg shadow-xl w-full max-w-xl max-h-[85vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border">
          <h2 className="text-lg font-semibold text-foreground">
            Create Model Profile
          </h2>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            <span aria-hidden="true">&times;</span>
          </Button>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-2 px-5 py-2 border-b border-border">
          {['Profile', 'Components', 'Review'].map((label, i) => (
            <React.Fragment key={label}>
              <span className={`text-sm font-medium ${i === step ? 'text-primary' : i < step ? 'text-muted-foreground' : 'text-muted-foreground/50'}`}>
                {i + 1}. {label}
              </span>
              {i < 2 && <span className="text-muted-foreground/40">→</span>}
            </React.Fragment>
          ))}
        </div>

        {/* Body — scrollable */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* ── Step 0: Profile ──────────────────────────────────────── */}
          {step === 0 && (
            <>
              <div className="flex flex-col gap-1.5">
                <label htmlFor="wiz-name" className="text-sm font-medium text-foreground">Name *</label>
                <input
                  id="wiz-name"
                  type="text"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="My LTX Profile"
                  className="h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  autoFocus
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label htmlFor="wiz-family" className="text-sm font-medium text-foreground">Family</label>
                <select
                  id="wiz-family"
                  value={family}
                  onChange={e => setFamily(e.target.value as ModelProfilePayload['family'])}
                  className="h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <option value="ltx-2">LTX 2</option>
                  <option value="ltx-2.3">LTX 2.3</option>
                  <option value="ltxv2">LTX v2</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              <div className="flex flex-col gap-1.5">
                <label htmlFor="wiz-source" className="text-sm font-medium text-foreground">Source</label>
                <select
                  id="wiz-source"
                  value={source}
                  onChange={e => setSource(e.target.value as ModelProfilePayload['source'])}
                  className="h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <option value="official">Official (monolithic + adapters)</option>
                  <option value="kijai">Kijai (split safetensors)</option>
                  <option value="quantstack">QuantStack</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              <div className="flex flex-col gap-1.5">
                <label htmlFor="wiz-notes" className="text-sm font-medium text-foreground">Notes</label>
                <textarea
                  id="wiz-notes"
                  value={notes}
                  onChange={e => setNotes(e.target.value)}
                  placeholder="Optional notes about this profile..."
                  rows={3}
                  className="rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none"
                />
              </div>
            </>
          )}

          {/* ── Step 1: Components ───────────────────────────────────── */}
          {step === 1 && (
            <>
              {/* Format selectors */}
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="wiz-tf" className="text-sm font-medium text-foreground">Transformer Format</label>
                  <select
                    id="wiz-tf"
                    value={transformerFormat}
                    onChange={e => setTransformerFormat(e.target.value)}
                    className="h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <option value="official_safetensors">Official Safetensors</option>
                    <option value="split_safetensors">Split Safetensors</option>
                    <option value="gguf">GGUF</option>
                  </select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="wiz-te" className="text-sm font-medium text-foreground">Text Encoder Format</label>
                  <select
                    id="wiz-te"
                    value={textEncoderFormat}
                    onChange={e => setTextEncoderFormat(e.target.value)}
                    className="h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <option value="api">API</option>
                    <option value="hf_folder">HuggingFace Folder</option>
                    <option value="safetensors">Safetensors</option>
                    <option value="gguf">GGUF</option>
                  </select>
                </div>
              </div>

              {/* Prefill from known models folder layout */}
              <div className="flex items-center gap-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePrefill}
                  disabled={prefillStatus === 'Scanning...'}
                >
                  {prefillStatus === 'Scanning...' ? (
                    <><Loader2 className="h-3 w-3 mr-1 animate-spin" /> Scanning...</>
                  ) : (
                    'Prefill from models folder'
                  )}
                </Button>
                {prefillStatus && prefillStatus !== 'Scanning...' && (
                  <span className="text-xs text-muted-foreground truncate max-w-80" title={prefillStatus}>
                    {prefillStatus}
                  </span>
                )}
              </div>

              {/* Dynamic component fields */}
              {COMPONENT_FIELDS.filter(f => visibleKeys.includes(f.key)).map(field => {
                const pickDir = field.pickDirectory ?? fieldPickDirectory(field.key, transformerFormat, textEncoderFormat)
                const filters = pickDir ? undefined : fieldFilters(field.key, transformerFormat, textEncoderFormat)
                return (
                  <ModelComponentPicker
                    key={field.key}
                    value={(components[field.key] as string) ?? ''}
                    onChange={v => updateComponent(field.key, v)}
                    label={field.label}
                    placeholder={`Path to ${field.label}...`}
                    dialogTitle={`Select ${field.label}`}
                    pickDirectory={pickDir}
                    filters={filters}
                  />
                )
              })}

              {visibleKeys.length === 0 && (
                <p className="text-sm text-muted-foreground py-2">
                  No component paths needed for this source.
                </p>
              )}

              {/* Official IC-LoRA Adapters (local/non-official profiles) */}
              {source !== 'official' && (
                <details className="mt-4 group" open={Object.keys(components.official_adapters ?? {}).length > 0}>
                  <summary className="cursor-pointer text-sm font-medium text-foreground hover:text-primary transition-colors">
                    Official IC-LoRA Adapters
                    {Object.keys(components.official_adapters ?? {}).length > 0 && (
                      <span className="ml-2 text-xs text-muted-foreground">
                        ({Object.keys(components.official_adapters!).length} set)
                      </span>
                    )}
                  </summary>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-2">
                    {OFFICIAL_ADAPTER_IDS.map(id => (
                      <div key={id} className="flex flex-col gap-0.5">
                        <label className="text-xs text-muted-foreground truncate">{OFFICIAL_ADAPTER_LABELS[id]}</label>
                        <input
                          type="text"
                          value={components.official_adapters?.[id] ?? ''}
                          onChange={e => {
                            const val = e.target.value
                            setComponents(prev => {
                              const current = { ...(prev.official_adapters ?? {}) }
                              if (val) current[id] = val
                              else delete current[id]
                              return { ...prev, official_adapters: current }
                            })
                          }}
                          placeholder={`Path to ${OFFICIAL_ADAPTER_LABELS[id]}...`}
                          className="h-7 rounded border border-border bg-background px-2 text-xs text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        />
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </>
          )}

          {/* ── Step 2: Review ────────────────────────────────────────── */}
          {step === 2 && (
            <>
              {/* Summary */}
              <div className="space-y-2 text-sm">
                <div><strong className="text-foreground">Name:</strong> <span className="text-muted-foreground">{name}</span></div>
                <div><strong className="text-foreground">Family:</strong> <span className="text-muted-foreground">{family}</span></div>
                <div><strong className="text-foreground">Source:</strong> <span className="text-muted-foreground">{source}</span></div>
                <div><strong className="text-foreground">Transformer Format:</strong> <span className="text-muted-foreground">{transformerFormat}</span></div>
                <div><strong className="text-foreground">Text Encoder Format:</strong> <span className="text-muted-foreground">{textEncoderFormat}</span></div>
                {notes && <div><strong className="text-foreground">Notes:</strong> <span className="text-muted-foreground">{notes}</span></div>}
                <div>
                  <strong className="text-foreground">Capabilities:</strong>{' '}
                  <span className="text-muted-foreground">{capabilities.length > 0 ? capabilities.join(', ') : '(none detected)'}</span>
                </div>
                {Object.entries(components).filter(([, v]) => v).length > 0 && (
                  <div>
                    <strong className="text-foreground">Components:</strong>
                    <ul className="list-disc list-inside text-muted-foreground mt-1">
                      {Object.entries(components)
                        .filter(([, v]) => typeof v === 'string')
                        .map(([k, v]) => (
                          <li key={k} className="truncate">{k}: {v as string}</li>
                        ))}
                      {Object.keys(components.official_adapters ?? {}).length > 0 && (
                        <li className="mt-1">
                          <strong>Official Adapters:</strong>
                          <ul className="list-disc list-inside ml-3 text-muted-foreground">
                            {Object.entries(components.official_adapters!).map(([id, p]) => (
                              <li key={id} className="truncate text-xs">{id}: {p}</li>
                            ))}
                          </ul>
                        </li>
                      )}
                    </ul>
                  </div>
                )}
              </div>

              {/* Actions */}
              {!createdProfileId && (
                <div className="flex flex-col gap-2 pt-2">
                  {createError && (
                    <div className="flex items-center gap-2 text-sm text-red-500">
                      <AlertTriangle className="h-4 w-4" />
                      {createError}
                    </div>
                  )}
                  <Button onClick={handleCreate} disabled={isCreating} className="w-full">
                    {isCreating ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      'Create Profile'
                    )}
                  </Button>
                </div>
              )}

              {createdProfileId && !validationResult && (
                <div className="flex flex-col gap-2 pt-2">
                  <div className="flex items-center gap-2 text-sm text-green-500">
                    <Check className="h-4 w-4" />
                    Profile created ✓
                  </div>
                  <Button onClick={handleValidate} disabled={isValidating} variant="outline" className="w-full">
                    {isValidating ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Validating...
                      </>
                    ) : (
                      'Validate Profile'
                    )}
                  </Button>
                </div>
              )}

              {validationResult && (
                <div className="flex flex-col gap-2 pt-2">
                  {validationResult.valid ? (
                    <div className="flex items-center gap-2 text-sm text-green-500">
                      <Check className="h-4 w-4" />
                      Validation passed ✓
                    </div>
                  ) : (
                    <div className="flex flex-col gap-1 text-sm text-red-500">
                      <div className="flex items-center gap-2 font-medium">
                        <AlertTriangle className="h-4 w-4" />
                        Validation issues:
                      </div>
                      <ul className="list-disc list-inside pl-2">
                        {(validationResult.issues ?? []).map((issue, i) => (
                          <li key={i}>{issue.field}: {issue.issue}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {activateError && (
                    <div className="flex items-center gap-2 text-sm text-red-500">
                      <AlertTriangle className="h-4 w-4" />
                      {activateError}
                    </div>
                  )}

                  <Button onClick={handleActivate} disabled={isActivating || generationRunning} className="w-full">
                    {isActivating ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Activating...
                      </>
                    ) : generationRunning ? (
                      'Wait for generation'
                    ) : (
                      'Activate Profile'
                    )}
                  </Button>
                </div>
              )}

              {/* Quick retry after activation fail */}
              {activateError && (
                <p className="text-xs text-muted-foreground">
                  Activation failed. You can close and retry from the Settings panel.
                </p>
              )}
            </>
          )}
        </div>

        {/* Footer — navigation */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-border">
          <Button
            variant="outline"
            onClick={step === 0 ? onClose : handlePrev}
            disabled={isCreating || isActivating}
          >
            {step === 0 ? (
              'Cancel'
            ) : (
              <>
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back
              </>
            )}
          </Button>

          {step < 2 && (
            <Button onClick={handleNext} disabled={!canGoNext}>
              Next
              <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
