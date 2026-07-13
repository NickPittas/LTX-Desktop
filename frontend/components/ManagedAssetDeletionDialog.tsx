import { useEffect, useId, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { AlertCircle, Check, Loader2, Trash2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { ManagedDeletionDialogState } from '@/hooks/use-managed-asset-deletion'

interface ManagedAssetDeletionDialogProps {
  dialogState: ManagedDeletionDialogState
  onProjectOnly: () => void
  onTrash: () => void
  onCancel: () => void
}

export function ManagedAssetDeletionDialog({
  dialogState,
  onProjectOnly,
  onTrash,
  onCancel,
}: ManagedAssetDeletionDialogProps) {
  const titleId = useId()
  const descId = useId()

  useEffect(() => {
    if (dialogState.status === 'idle') return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (dialogState.status === 'trashing') return
        event.preventDefault()
        onCancel()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [dialogState.status, onCancel])

  if (dialogState.status === 'idle') return null

  let title = 'Delete asset'
  let icon: ReactNode = <Trash2 className="h-5 w-5 text-zinc-400" />
  let body: ReactNode = null
  let footer: ReactNode = null

  switch (dialogState.status) {
    case 'analyzing':
      title = 'Checking files…'
      icon = <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
      body = (
        <p id={descId} className="text-sm text-zinc-300">
          Analyzing references so only unshared, app-managed files are offered for
          Trash.
        </p>
      )
      footer = (
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
      )
      break

    case 'confirm': {
      const { eligiblePaths, canTrash } = dialogState
      title = 'Delete from project?'
      icon = <Trash2 className="h-5 w-5 text-zinc-400" />
      body = (
        <div className="space-y-3">
          <p id={descId} className="text-sm text-zinc-300">
            {canTrash
              ? `${eligiblePaths.length} managed file${
                  eligiblePaths.length === 1 ? '' : 's'
                } can be moved to the Trash.`
              : 'No app-managed files are eligible for Trash. Imported or shared files will stay on disk.'}
          </p>
          {eligiblePaths.length > 0 && (
            <ul
              className="max-h-40 overflow-auto rounded-lg bg-zinc-800/50 p-2 text-xs text-zinc-400 space-y-1"
              aria-label="Files that will be moved to Trash"
            >
              {eligiblePaths.map((path) => (
                <li key={path} className="break-all">
                  {path}
                </li>
              ))}
            </ul>
          )}
        </div>
      )
      footer = (
        <>
          <Button variant="outline" onClick={onCancel} autoFocus>
            Cancel
          </Button>
          <Button variant="secondary" onClick={onProjectOnly}>
            Remove from project
          </Button>
          {canTrash && (
            <Button
              variant="destructive"
              onClick={onTrash}
              className="gap-2"
            >
              <Trash2 className="h-4 w-4" />
              Move to Trash
            </Button>
          )}
        </>
      )
      break
    }

    case 'preflight-error':
      title = 'Cannot delete safely'
      icon = <AlertCircle className="h-5 w-5 text-red-400" />
      body = (
        <div className="space-y-3">
          <p id={descId} className="text-sm text-zinc-300">
            {dialogState.error}
          </p>
          <p className="text-xs text-zinc-500">
            You can still remove the item from the project without moving files
            to Trash.
          </p>
        </div>
      )
      footer = (
        <>
          <Button variant="outline" onClick={onCancel} autoFocus>
            Cancel
          </Button>
          <Button variant="secondary" onClick={onProjectOnly}>
            Remove from project
          </Button>
        </>
      )
      break

    case 'trashing':
      title = 'Moving to Trash…'
      icon = <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
      body = (
        <p id={descId} className="text-sm text-zinc-300">
          Moving eligible files to the Trash.
        </p>
      )
      footer = null
      break

    case 'trash-complete':
      title = 'Moved to Trash'
      icon = <Check className="h-5 w-5 text-green-400" />
      body = (
        <div className="space-y-3">
          <p id={descId} className="text-sm text-zinc-300">
            {dialogState.trashedPaths.length} file
            {dialogState.trashedPaths.length === 1 ? '' : 's'} moved to Trash.
          </p>
          {dialogState.trashedPaths.length > 0 && (
            <ul
              className="max-h-32 overflow-auto rounded-lg bg-zinc-800/50 p-2 text-xs text-zinc-400 space-y-1"
              aria-label="Trashed files"
            >
              {dialogState.trashedPaths.map((path) => (
                <li key={path} className="break-all">
                  {path}
                </li>
              ))}
            </ul>
          )}
        </div>
      )
      footer = (
        <Button variant="outline" onClick={onCancel} autoFocus>
          Done
        </Button>
      )
      break

    case 'trash-partial':
      title = 'Trash partially failed'
      icon = <AlertCircle className="h-5 w-5 text-amber-400" />
      body = (
        <div className="space-y-3">
          <p id={descId} className="text-sm text-zinc-300">
            {dialogState.trashedPaths.length === 0
              ? 'No files were moved to Trash, but the asset was removed from the project.'
              : 'Some files were moved to Trash, but others could not be.'}
          </p>
          {dialogState.trashedPaths.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-1">
                Trashed
              </p>
              <ul className="max-h-24 overflow-auto rounded-lg bg-zinc-800/50 p-2 text-xs text-zinc-400 space-y-1">
                {dialogState.trashedPaths.map((path) => (
                  <li key={path} className="break-all">
                    {path}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {dialogState.failedPaths.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-1">
                Failed
              </p>
              <ul className="max-h-24 overflow-auto rounded-lg bg-red-500/10 p-2 text-xs text-red-300 space-y-1">
                {dialogState.failedPaths.map(({ path, reason }) => (
                  <li key={path} className="break-all">
                    {path}: {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )
      footer = (
        <Button variant="outline" onClick={onCancel} autoFocus>
          Done
        </Button>
      )
      break

    case 'post-trash-error':
      title = 'Files may already be in Trash'
      icon = <AlertCircle className="h-5 w-5 text-amber-400" />
      body = (
        <div className="space-y-3">
          <p id={descId} className="text-sm text-zinc-300">Files may already be in the OS Trash, but the project could not be updated safely.</p>
          <p className="text-xs text-red-300 break-all">{dialogState.error}</p>
          {dialogState.trashedPaths.length > 0 && <PathList label="Moved to Trash" paths={dialogState.trashedPaths} />}
          {dialogState.failedPaths.length > 0 && <FailedPathList paths={dialogState.failedPaths} />}
        </div>
      )
      footer = <Button variant="outline" onClick={onCancel} autoFocus>Done</Button>
      break

    default:
      return null
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={dialogState.status === 'trashing' ? undefined : onCancel}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descId}
        className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-[480px] max-h-[calc(100vh-2rem)] max-h-[calc(100dvh-2rem)] flex flex-col overflow-hidden"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            {icon}
            <h2 id={titleId} className="text-base font-semibold text-zinc-100">
              {title}
            </h2>
          </div>
          <button
            type="button"
            onClick={dialogState.status === 'trashing' ? undefined : onCancel}
            disabled={dialogState.status === 'trashing'}
            aria-label="Close"
            className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-6 py-5 overflow-y-auto">{body}</div>

        {footer && (
          <div className="px-6 py-4 border-t border-zinc-800 flex flex-wrap items-center justify-end gap-3">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body
  )
}

function PathList({ label, paths }: { label: string; paths: string[] }) {
  return <div><p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-1">{label}</p><ul className="max-h-24 overflow-auto rounded-lg bg-zinc-800/50 p-2 text-xs text-zinc-400 space-y-1">{paths.map(path => <li key={path} className="break-all">{path}</li>)}</ul></div>
}

function FailedPathList({ paths }: { paths: Array<{ path: string; reason: string }> }) {
  return <div><p className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-1">Failed</p><ul className="max-h-24 overflow-auto rounded-lg bg-red-500/10 p-2 text-xs text-red-300 space-y-1">{paths.map(({ path, reason }) => <li key={path} className="break-all">{path}: {reason}</li>)}</ul></div>
}
