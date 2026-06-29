import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  ExternalLink,
  FileQuestion,
  FileWarning,
  FolderOpen,
  HelpCircle,
  Loader2,
  Lock,
  RefreshCw,
  Search,
  X,
} from 'lucide-react'
import { ApiClient } from '../lib/api-client'
import { useModelLibrary } from '../hooks/use-model-library'
import { Button } from './ui/button'
import { Progress } from './ui/progress'
import { Tooltip } from './ui/tooltip'
import type {
  ArtifactKind,
  ArtifactStatus,
  ArtifactSupportStatus,
  CatalogSection,
  ModelLibraryArtifact,
  ModelDownloadProgressResponse,
  ContentPieceId,
} from '../types/model-library'

interface ModelLibraryPanelProps {
  isOpen: boolean
}

type StatusFilter = ArtifactStatus | 'gated' | 'unvalidated' | 'all'

const KIND_LABELS: Record<ArtifactKind, string> = {
  diffusion_model: 'Diffusion Model',
  vae: 'VAE',
  text_encoder: 'Text Encoder',
  gguf: 'GGUF',
  upscaler: 'Upscaler',
  control_adapter: 'Control Adapter',
  lora: 'LoRA',
  scene_embeddings: 'Scene Embeddings',
  depth_processor: 'Depth Processor',
  pose_processor: 'Pose Processor',
  person_detector: 'Person Detector',
  image_gen_model: 'Image Generation',
}

const SECTION_ORDER: CatalogSection[] = ['full', 'kijai', 'gguf', 'addons']

const SECTION_LABELS: Record<CatalogSection, { label: string; description: string }> = {
  full: { label: 'Full', description: 'Core model files needed for local workflows' },
  kijai: { label: 'Kijai', description: 'Alternative Kijai FP8 builds' },
  gguf: { label: 'GGUF', description: 'Quantized GGUF model variants' },
  addons: { label: 'Add-ons & Controls', description: 'Optional adapters, controls, and utility models' },
}

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || bytes === 0) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex++
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

function formatScannedAt(iso: string): string {
  try {
    const date = new Date(iso)
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    })
  } catch {
    return iso
  }
}

function statusOrder(status: ArtifactStatus): number {
  switch (status) {
    case 'missing':
      return 0
    case 'wrong_folder_usable':
      return 1
    case 'duplicate':
      return 2
    case 'installed':
      return 3
    default:
      return 4
  }
}

function normalizePath(p: string): string {
  return p.replace(/\\/g, '/').replace(/\/+/g, '/')
}

function getExpectedAbsolutePath(modelsDir: string, artifact: ModelLibraryArtifact): string {
  return normalizePath(`${modelsDir}/${artifact.canonical_relative_path}`)
}

function hasCanonicalCopy(modelsDir: string, artifact: ModelLibraryArtifact): boolean {
  if (!artifact.absolute_paths?.length) return false
  const expected = getExpectedAbsolutePath(modelsDir, artifact)
  return artifact.absolute_paths.some((p) => normalizePath(p) === expected)
}

function resolveDownloadAction(
  artifact: ModelLibraryArtifact,
  modelsDir: string,
): {
  label: string
  disabled: boolean
  cpId: ContentPieceId | null
  kind: 'primary' | 'muted' | 'info'
  tooltip: string
} {
  const cpId = artifact.cp_id ?? null

  if (!cpId) {
    return {
      label: 'No download source',
      disabled: true,
      cpId: null,
      kind: 'muted',
      tooltip: 'There is no known download source for this file.',
    }
  }

  if (artifact.support_status === 'gated' || artifact.gated) {
    return {
      label: 'Gated',
      disabled: true,
      cpId,
      kind: 'info',
      tooltip: 'Accept the upstream license on Hugging Face to download this file.',
    }
  }

  if (artifact.status === 'installed') {
    return {
      label: 'Installed',
      disabled: true,
      cpId,
      kind: 'muted',
      tooltip: 'This file is already in the expected location.',
    }
  }

  if (artifact.status === 'duplicate') {
    if (hasCanonicalCopy(modelsDir, artifact)) {
      return {
        label: 'Canonical copy present',
        disabled: true,
        cpId,
        kind: 'muted',
        tooltip: 'The canonical copy is already present; extra copies were ignored.',
      }
    }
    return {
      label: 'Download canonical copy',
      disabled: false,
      cpId,
      kind: 'primary',
      tooltip: 'Download the canonical copy to the expected location.',
    }
  }

  if (artifact.status === 'wrong_folder_usable') {
    return {
      label: 'Download canonical copy',
      disabled: false,
      cpId,
      kind: 'primary',
      tooltip: 'This file works from its current location, but it is not in the expected folder.',
    }
  }

  if (artifact.support_status === 'unvalidated') {
    return {
      label: 'Download (unvalidated)',
      disabled: false,
      cpId,
      kind: 'primary',
      tooltip: 'This file has not been validated for this version of the app. Download at your own risk.',
    }
  }

  return {
    label: 'Download',
    disabled: false,
    cpId,
    kind: 'primary',
    tooltip: 'Download this file to the expected location.',
  }
}

function StatusBadge({ status, count }: { status: ArtifactStatus; count?: number }) {
  const config: Record<
    ArtifactStatus,
    { label: string; icon: React.ReactNode; classes: string; tooltip: string }
  > = {
    installed: {
      label: 'Installed',
      icon: <CheckCircle2 className="h-3 w-3" />,
      classes: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
      tooltip: 'File is present in the expected location.',
    },
    missing: {
      label: 'Missing',
      icon: <AlertCircle className="h-3 w-3" />,
      classes: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
      tooltip: 'File is not present.',
    },
    wrong_folder_usable: {
      label: 'Wrong folder',
      icon: <FolderOpen className="h-3 w-3" />,
      classes: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20',
      tooltip: 'File was found outside the expected path. It is usable from its current location.',
    },
    duplicate: {
      label: 'Duplicate',
      icon: <FileWarning className="h-3 w-3" />,
      classes: 'bg-blue-500/15 text-blue-400 border-blue-500/20',
      tooltip: 'Multiple copies of this file were found.',
    },
  }

  const c = config[status]
  return (
    <Tooltip content={c.tooltip}>
      <span
        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border ${c.classes}`}
      >
        {c.icon}
        {c.label}
        {typeof count === 'number' && count > 0 && (
          <span className="opacity-80">{count}</span>
        )}
      </span>
    </Tooltip>
  )
}

function SupportBadge({ status }: { status: ArtifactSupportStatus }) {
  if (status === 'not_applicable') return null

  const config = {
    supported: {
      label: 'Supported',
      classes: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/15',
      icon: <CheckCircle2 className="h-3 w-3" />,
      tooltip: 'This file is supported for current workflows.',
    },
    gated: {
      label: 'Gated',
      classes: 'bg-purple-500/15 text-purple-400 border-purple-500/25',
      icon: <Lock className="h-3 w-3" />,
      tooltip: 'Accept the upstream license on Hugging Face to download this file.',
    },
    unvalidated: {
      label: 'Unvalidated',
      classes: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/25',
      icon: <HelpCircle className="h-3 w-3" />,
      tooltip: 'This file has not been validated for this version of the app.',
    },
  }

  const c = config[status]
  return (
    <Tooltip content={c.tooltip}>
      <span
        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border ${c.classes}`}
      >
        {c.icon}
        {c.label}
      </span>
    </Tooltip>
  )
}

function ArtifactRow({
  artifact,
  modelsDir,
  onDownload,
  isDownloading,
}: {
  artifact: ModelLibraryArtifact
  modelsDir: string
  onDownload: (cpId: ContentPieceId) => void
  isDownloading: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  const action = resolveDownloadAction(artifact, modelsDir)
  const actualPaths = artifact.absolute_paths ?? []
  const expectedPath = getExpectedAbsolutePath(modelsDir, artifact)

  return (
    <div className="rounded-lg border border-zinc-700/50 bg-zinc-800/30 hover:bg-zinc-800/50 transition-colors">
      <div className="flex flex-col lg:flex-row lg:items-center gap-3 p-3">
        <div className="flex-1 min-w-0 space-y-0.5">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-zinc-100 truncate">{artifact.filename}</span>
            <StatusBadge status={artifact.status} />
            <SupportBadge status={artifact.support_status} />
          </div>
          <div className="text-[10px] text-zinc-500 truncate">
            {artifact.component_role}
            {artifact.artifact_kind && artifact.component_role !== KIND_LABELS[artifact.artifact_kind] && (
              <> · {KIND_LABELS[artifact.artifact_kind]}</>
            )}
          </div>
        </div>

        <div className="lg:w-2/5 min-w-0 text-xs text-zinc-400 space-y-0.5">
          <div className="truncate">
            <span className="text-zinc-500">Expected location:</span>{' '}
            <Tooltip content={expectedPath}>
              <span className="font-mono text-zinc-500">{expectedPath}</span>
            </Tooltip>
          </div>
          {artifact.status !== 'missing' && actualPaths.length > 0 && (
            <div className="truncate">
              <span className="text-zinc-500">Found at:</span>{' '}
              <Tooltip content={actualPaths.join('\n')}>
                <span className="font-mono text-zinc-500">{actualPaths[0]}</span>
              </Tooltip>
              {actualPaths.length > 1 && (
                <span className="text-zinc-500"> +{actualPaths.length - 1} more</span>
              )}
            </div>
          )}
          {artifact.status === 'wrong_folder_usable' && (
            <p className="text-[10px] text-yellow-500/80">Usable from current location.</p>
          )}
          {artifact.status === 'duplicate' && actualPaths.length > 1 && (
            <p className="text-[10px] text-blue-400/80">{actualPaths.length} copies found.</p>
          )}
        </div>

        <div className="lg:w-32 text-xs text-zinc-400 text-left lg:text-right shrink-0">
          <div>{formatBytes(artifact.size_bytes)}</div>
          <div className="text-zinc-500">/ {formatBytes(artifact.expected_size_bytes)}</div>
        </div>

        <div className="flex items-center gap-2 lg:w-48 shrink-0">
          {action.disabled ? (
            <Tooltip content={action.tooltip}>
              <span className="inline-flex items-center justify-center flex-1 px-3 py-1.5 rounded-md text-xs font-medium bg-zinc-800 text-zinc-500 border border-zinc-700 cursor-not-allowed">
                {action.label}
              </span>
            </Tooltip>
          ) : (
            <Button
              size="sm"
              variant={action.kind === 'primary' ? 'default' : 'outline'}
              disabled={isDownloading || action.disabled}
              onClick={() => {
                if (action.cpId) onDownload(action.cpId)
              }}
              className="flex-1 text-xs"
            >
              <Download className="h-3 w-3 mr-1.5" />
              {action.label}
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            aria-label={expanded ? 'Collapse details' : 'Expand details'}
            aria-expanded={expanded}
            onClick={() => setExpanded((v) => !v)}
            className="h-8 w-8 p-0 text-zinc-400 hover:text-white hover:bg-zinc-700"
          >
            {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {expanded && (
        <div className="px-3 pb-3 pt-0 border-t border-zinc-700/30">
          <div className="pt-3 space-y-2 text-xs text-zinc-400">
            {artifact.repo_id && (
              <div className="flex items-center gap-2">
                <span className="text-zinc-500 shrink-0">Repo:</span>
                <span className="font-mono truncate">{artifact.repo_id}</span>
                {artifact.source_url && (
                  <a
                    href={artifact.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-blue-400 hover:text-blue-300 shrink-0"
                  >
                    Source <ExternalLink className="h-3 w-3 ml-0.5" />
                  </a>
                )}
              </div>
            )}
            {artifact.cp_id && (
              <div className="flex items-center gap-2">
                <span className="text-zinc-500 shrink-0">Content piece:</span>
                <span className="font-mono truncate">{artifact.cp_id}</span>
              </div>
            )}
            {artifact.adapter_id && (
              <div className="flex items-center gap-2">
                <span className="text-zinc-500 shrink-0">Adapter:</span>
                <span className="font-mono truncate">{artifact.adapter_id}</span>
              </div>
            )}
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 shrink-0">Canonical relative path:</span>
              <span className="font-mono truncate">{artifact.canonical_relative_path}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 shrink-0">Expected location:</span>
              <span className="font-mono truncate">{expectedPath}</span>
            </div>
            {artifact.preferred_path && (
              <div className="flex items-center gap-2">
                <span className="text-zinc-500 shrink-0">Preferred path:</span>
                <span className="font-mono truncate">{artifact.preferred_path}</span>
              </div>
            )}
            {actualPaths.length > 0 && (
              <div className="space-y-1">
                <span className="text-zinc-500">All found paths:</span>
                <ul className="list-disc list-inside font-mono text-zinc-500">
                  {actualPaths.map((p, i) => (
                    <li key={i} className="truncate">{p}</li>
                  ))}
                </ul>
              </div>
            )}
            {artifact.notes && (
              <div className="text-zinc-400 bg-zinc-900/50 rounded p-2">{artifact.notes}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function SectionHeader({ section, count }: { section: CatalogSection; count: number }) {
  const info = SECTION_LABELS[section]
  return (
    <div className="flex items-center justify-between py-2 px-3 border-b border-zinc-700/50 bg-zinc-900/80">
      <div className="flex flex-col gap-0.5">
        <span className="text-xs font-semibold text-white">{info.label}</span>
        <span className="text-[10px] text-zinc-500">{info.description}</span>
      </div>
      <span className="text-[10px] text-zinc-500">{count} {count === 1 ? 'file' : 'files'}</span>
    </div>
  )
}

function GroupHeader({ kind, count }: { kind: ArtifactKind; count: number }) {
  return (
    <div className="sticky top-0 z-10 flex items-center justify-between py-2 px-3 bg-zinc-900/95 backdrop-blur border-y border-zinc-800">
      <span className="text-xs font-semibold text-zinc-300">{KIND_LABELS[kind]}</span>
      <span className="text-[10px] text-zinc-500">{count} {count === 1 ? 'file' : 'files'}</span>
    </div>
  )
}

function CollapsiblePathList({
  title,
  items,
  icon,
}: {
  title: string
  items: { absolute_path: string; relative_path?: string; size_bytes?: number }[]
  icon: React.ReactNode
}) {
  const [open, setOpen] = useState(false)
  if (items.length === 0) return null

  return (
    <div className="border border-zinc-700/50 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 bg-zinc-800/30 hover:bg-zinc-800/50 text-xs text-zinc-400 transition-colors"
      >
        <span className="flex items-center gap-2">
          {icon}
          {title} ({items.length})
        </span>
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
      </button>
      {open && (
        <ul className="divide-y divide-zinc-700/30">
          {items.map((item, i) => (
            <li key={i} className="px-3 py-2 text-[10px] text-zinc-500 font-mono truncate flex items-center justify-between gap-2"
            >
              <Tooltip content={item.absolute_path}>
                <span className="truncate">{item.relative_path ?? item.absolute_path}</span>
              </Tooltip>
              {typeof item.size_bytes === 'number' && (
                <span className="shrink-0 text-zinc-600">{formatBytes(item.size_bytes)}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function SkeletonRows({ count = 6 }: { count?: number }) {
  return (
    <div className="space-y-2 animate-pulse">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="h-16 rounded-lg bg-zinc-800/50 border border-zinc-700/30">
          <div className="h-full flex items-center gap-3 px-3">
            <div className="h-4 w-1/3 rounded bg-zinc-700"></div>
            <div className="h-4 w-16 rounded bg-zinc-700"></div>
            <div className="ml-auto h-8 w-24 rounded bg-zinc-700"></div>
          </div>
        </div>
      ))}
    </div>
  )
}

export function ModelLibraryPanel({ isOpen }: ModelLibraryPanelProps) {
  const { catalog, isLoading, errorMessage, refresh } = useModelLibrary(isOpen)
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<ModelDownloadProgressResponse | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [isCancelling, setIsCancelling] = useState(false)

  const isDownloading = !!downloadSessionId

  const counts = useMemo(() => {
    const empty: Record<ArtifactStatus | 'gated' | 'unvalidated', number> = {
      installed: 0,
      missing: 0,
      wrong_folder_usable: 0,
      duplicate: 0,
      gated: 0,
      unvalidated: 0,
    }
    if (!catalog?.artifacts) return empty
    return catalog.artifacts.reduce((acc, a) => {
      acc[a.status]++
      if (a.support_status === 'gated' || a.gated) acc.gated++
      if (a.support_status === 'unvalidated') acc.unvalidated++
      return acc
    }, empty)
  }, [catalog])

  const groupedArtifacts = useMemo(() => {
    if (!catalog?.artifacts) return new Map<CatalogSection, Map<ArtifactKind, ModelLibraryArtifact[]>>()
    const query = search.trim().toLowerCase()
    const filtered = catalog.artifacts.filter((a) => {
      const matchesStatus =
        statusFilter === 'all' ||
        (statusFilter === 'gated'
          ? a.support_status === 'gated' || a.gated
          : statusFilter === 'unvalidated'
            ? a.support_status === 'unvalidated'
            : a.status === statusFilter)
      const matchesSearch =
        !query ||
        a.filename.toLowerCase().includes(query) ||
        (a.repo_id?.toLowerCase().includes(query) ?? false) ||
        (a.cp_id?.toLowerCase().includes(query) ?? false)
      return matchesStatus && matchesSearch
    })
    const sorted = [...filtered].sort((a, b) => {
      const sectionOrder = SECTION_ORDER.indexOf(a.section) - SECTION_ORDER.indexOf(b.section)
      if (sectionOrder !== 0) return sectionOrder
      const kindOrder = a.artifact_kind.localeCompare(b.artifact_kind)
      if (kindOrder !== 0) return kindOrder
      const statusDiff = statusOrder(a.status) - statusOrder(b.status)
      if (statusDiff !== 0) return statusDiff
      return a.filename.localeCompare(b.filename)
    })
    const groups = new Map<CatalogSection, Map<ArtifactKind, ModelLibraryArtifact[]>>()
    for (const artifact of sorted) {
      const section = artifact.section
      let sectionMap = groups.get(section)
      if (!sectionMap) {
        sectionMap = new Map<ArtifactKind, ModelLibraryArtifact[]>()
        groups.set(section, sectionMap)
      }
      const list = sectionMap.get(artifact.artifact_kind) ?? []
      list.push(artifact)
      sectionMap.set(artifact.artifact_kind, list)
    }
    return groups
  }, [catalog, search, statusFilter])

  const handleStartDownload = useCallback(async (cpId: ContentPieceId) => {
    setDownloadError(null)
    setDownloadProgress(null)
    const result = await ApiClient.startModelDownload({ type: 'download', cp_ids: [cpId] })
    if (!result.ok) {
      setDownloadError(result.error.message)
      return
    }
    if (result.data.status === 'started') {
      setDownloadSessionId(result.data.sessionId)
    }
  }, [])

  const handleCancelDownload = useCallback(async () => {
    setIsCancelling(true)
    const result = await ApiClient.cancelModelDownload()
    if (!result.ok) {
      setDownloadError(result.error.message)
      setIsCancelling(false)
      return
    }
    if (result.data.status === 'no_active_download') {
      setIsCancelling(false)
      setDownloadSessionId(null)
      setDownloadProgress(null)
    }
    // 'cancelling' will resolve via the progress poll.
  }, [])

  useEffect(() => {
    if (!downloadSessionId) return

    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!result.ok) return
      setDownloadProgress(result.data)

      if (result.data.status === 'complete' || result.data.status === 'error' || result.data.status === 'cancelled') {
        setDownloadSessionId(null)
        setIsCancelling(false)
        setDownloadProgress(null)
        if (result.data.status === 'error') {
          setDownloadError(result.data.error ?? 'Download failed')
        }
        await refresh()
      }
    }

    void poll()
    const interval = setInterval(() => { void poll() }, 1000)
    return () => clearInterval(interval)
  }, [downloadSessionId, refresh])

  const renderProgressBanner = () => {
    if (!downloadSessionId) return null
    const currentFile = downloadProgress?.status === 'downloading'
      ? downloadProgress.current_downloading_file
      : null
    const progress = downloadProgress?.status === 'downloading'
      ? downloadProgress.total_progress
      : 0
    const speed = downloadProgress?.status === 'downloading'
      ? downloadProgress.speed_bytes_per_sec
      : 0

    return (
      <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 text-blue-300">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{isCancelling ? 'Cancelling download…' : 'Downloading model files…'}</span>
          </div>
          <Button
            size="sm"
            variant="ghost"
            disabled={isCancelling}
            onClick={handleCancelDownload}
            className="h-7 px-2 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10"
          >
            <X className="h-3 w-3 mr-1" />
            Cancel
          </Button>
        </div>
        {currentFile && (
          <div className="text-[10px] text-zinc-400 truncate">Current file: <span className="font-mono">{currentFile}</span></div>
        )}
        <Progress value={progress} max={100} />
        <div className="flex justify-between text-[10px] text-zinc-500">
          <span>{Math.round(progress)}% complete</span>
          {speed > 0 && <span>{formatBytes(speed)}/s</span>}
        </div>
      </div>
    )
  }

  const header = (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-0.5 min-w-0">
          <h3 className="text-sm font-semibold text-white">Model Library</h3>
          {catalog && (
            <>
              <div className="flex items-center gap-2 text-xs text-zinc-400">
                <FolderOpen className="h-3.5 w-3.5 shrink-0" />
                <Tooltip content={catalog.models_dir}>
                  <span className="truncate max-w-[280px] sm:max-w-md font-mono">{catalog.models_dir}</span>
                </Tooltip>
              </div>
              <div className="text-[10px] text-zinc-500">
                Scanned {formatScannedAt(catalog.scanned_at)}
              </div>
            </>
          )}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => void refresh()}
          disabled={isLoading}
          className="shrink-0 border-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-800"
        >
          {isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : <RefreshCw className="h-3.5 w-3.5 mr-1.5" />}
          Rescan
        </Button>
      </div>

      {renderProgressBanner()}

      {downloadError && (
        <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          {downloadError}
        </div>
      )}

      {errorMessage && !isLoading && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <div className="flex-1">{errorMessage}</div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void refresh()}
            className="h-6 px-2 text-xs text-amber-300 hover:text-amber-200 hover:bg-amber-500/10"
          >
            Retry
          </Button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { key: 'installed', label: 'Installed', count: counts.installed },
            { key: 'missing', label: 'Missing', count: counts.missing },
            { key: 'wrong_folder_usable', label: 'Wrong folder', count: counts.wrong_folder_usable },
            { key: 'duplicate', label: 'Duplicate', count: counts.duplicate },
            { key: 'gated', label: 'Gated', count: counts.gated },
            { key: 'unvalidated', label: 'Unvalidated', count: counts.unvalidated },
          ] as const
        ).map((chip) => {
          const active = statusFilter === chip.key
          return (
            <button
              key={chip.key}
              type="button"
              onClick={() => setStatusFilter((prev) => (prev === chip.key ? 'all' : chip.key))}
              className={`px-2 py-1 rounded-full text-[10px] font-medium border transition-colors ${
                active
                  ? 'bg-zinc-700 text-white border-zinc-600'
                  : 'bg-zinc-800/50 text-zinc-400 border-zinc-700 hover:bg-zinc-800 hover:text-zinc-300'
              }`}
            >
              {chip.label}
              {chip.count > 0 && <span className="ml-1 text-zinc-500">{chip.count}</span>}
            </button>
          )
        })}
      </div>

      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by filename, repo, or content piece"
          className="w-full pl-8 pr-3 py-1.5 rounded-md bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>
    </div>
  )

  const renderEmptyState = () => (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <FileQuestion className="h-10 w-10 text-zinc-600 mb-3" />
      <p className="text-sm text-zinc-400">No models match your filters.</p>
      <Button
        size="sm"
        variant="ghost"
        onClick={() => { setSearch(''); setStatusFilter('all') }}
        className="mt-2 text-xs text-blue-400 hover:text-blue-300"
      >
        Clear filters
      </Button>
    </div>
  )

  return (
    <div className="space-y-4">
      {header}

      {isLoading && !catalog ? (
        <SkeletonRows />
      ) : errorMessage && !catalog ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <AlertCircle className="h-10 w-10 text-amber-500/60 mb-3" />
          <p className="text-sm text-zinc-400">Couldn’t load the model library.</p>
          <p className="text-xs text-zinc-500 mt-1">{errorMessage}</p>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void refresh()}
            className="mt-4 border-zinc-700 text-zinc-300 hover:text-white"
          >
            Retry
          </Button>
        </div>
      ) : groupedArtifacts.size === 0 ? (
        catalog?.artifacts?.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <FolderOpen className="h-10 w-10 text-zinc-600 mb-3" />
            <p className="text-sm text-zinc-400">No models found in the library.</p>
            <p className="text-xs text-zinc-500 mt-1 max-w-sm">
              Place model files in the models folder above, then rescan.
            </p>
          </div>
        ) : (
          renderEmptyState()
        )
      ) : (
        <div className="space-y-6">
          {Array.from(groupedArtifacts.entries()).map(([section, sectionArtifacts]) => {
            const sectionCount = Array.from(sectionArtifacts.values()).reduce(
              (sum, list) => sum + list.length,
              0,
            )
            return (
              <div key={section} className="space-y-2">
                <SectionHeader section={section} count={sectionCount} />
                <div className="space-y-4">
                  {Array.from(sectionArtifacts.entries()).map(([kind, artifacts]) => (
                    <div key={`${section}-${kind}`} className="space-y-1">
                      <GroupHeader kind={kind} count={artifacts.length} />
                      <div className="space-y-1.5">
                        {artifacts.map((artifact) => (
                          <ArtifactRow
                            key={`${artifact.filename}-${artifact.canonical_relative_path}`}
                            artifact={artifact}
                            modelsDir={catalog?.models_dir ?? ''}
                            onDownload={handleStartDownload}
                            isDownloading={isDownloading}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {catalog && (
        <div className="space-y-2 pt-2">
          <CollapsiblePathList
            title="Unrecognized files"
            items={catalog.unknown_files ?? []}
            icon={<FileQuestion className="h-3.5 w-3.5" />}
          />
          <CollapsiblePathList
            title="Partial downloads"
            items={catalog.partial_files ?? []}
            icon={<FileWarning className="h-3.5 w-3.5" />}
          />
        </div>
      )}
    </div>
  )
}
