import { useCallback, useEffect, useRef, useState } from 'react'
import type { Project } from '@/types/project-model'
import { readAllProjectsForReferenceAnalysis } from '@/lib/project-storage'
import {
  buildManagedDeletionPlan,
  type DeletionTarget,
} from '@/lib/managed-file-references'
import type { EditorModel } from '@/views/editor/editor-state'

export type { DeletionTarget }

export interface ManagedDeletionProjectStore {
  kind: 'project'
  project: Project
  removeTargets: (targets: DeletionTarget[]) => void
}

export interface ManagedDeletionEditorStore {
  kind: 'editor'
  project: Project
  editorModel: EditorModel
  removeTargets: (targets: DeletionTarget[]) => void
  clearHistory: () => void
}

export type ManagedDeletionStore = ManagedDeletionProjectStore | ManagedDeletionEditorStore

export type ManagedDeletionDialogState =
  | { status: 'idle' }
  | { status: 'analyzing'; targets: DeletionTarget[] }
  | {
      status: 'confirm'
      targets: DeletionTarget[]
      eligiblePaths: string[]
      canTrash: boolean
    }
  | { status: 'preflight-error'; targets: DeletionTarget[]; error: string }
  | { status: 'trashing'; targets: DeletionTarget[]; eligiblePaths: string[] }
  | {
      status: 'trash-complete'
      trashedPaths: string[]
      failedPaths: Array<{ path: string; reason: string }>
    }
  | {
      status: 'trash-partial'
      trashedPaths: string[]
      failedPaths: Array<{ path: string; reason: string }>
    }
  | {
      status: 'post-trash-error'
      trashedPaths: string[]
      failedPaths: Array<{ path: string; reason: string }>
      error: string
    }

export interface UseManagedAssetDeletionInput {
  projectId: string
  store: ManagedDeletionStore
}

export interface UseManagedAssetDeletionOutput {
  dialogState: ManagedDeletionDialogState
  requestDeletion: (targets: DeletionTarget[]) => Promise<void>
  confirmProjectOnly: () => Promise<void>
  confirmTrash: () => Promise<void>
  cancelDeletion: () => void
}

function joinPath(left: string, right: string): string {
  return `${left.replace(/[\\/]+$/, '')}/${right}`
}

export function useManagedAssetDeletion({
  projectId,
  store,
}: UseManagedAssetDeletionInput): UseManagedAssetDeletionOutput {
  const [dialogState, setDialogState] = useState<ManagedDeletionDialogState>({
    status: 'idle',
  })
  const dialogStateRef = useRef(dialogState)
  dialogStateRef.current = dialogState
  const generationRef = useRef(0)
  const projectIdRef = useRef(projectId)
  const storeRef = useRef(store)
  projectIdRef.current = projectId
  storeRef.current = store
  useEffect(() => () => { generationRef.current += 1 }, [])

  const buildPlan = useCallback(async (targets: DeletionTarget[]) => {
    const assetsRoot = await window.electronAPI.getProjectAssetsPath()
    const latestStore = storeRef.current
    const latestProjectId = projectIdRef.current
    if (latestProjectId !== latestStore.project.id) throw new Error('Project changed while preparing deletion')
    const plan = buildManagedDeletionPlan({
      persistedProjects: readAllProjectsForReferenceAnalysis(),
      currentProject: latestStore.project,
      editorSnapshot: latestStore.kind === 'editor'
        ? { projectId: latestProjectId, model: latestStore.editorModel }
        : null,
      targets,
      projectManagedDirectory: joinPath(assetsRoot, latestProjectId),
      generationRoot: joinPath(assetsRoot, '.ltx-generations'),
    })
    return { plan, store: latestStore, projectId: latestProjectId }
  }, [])

  const requestDeletion = useCallback(
    async (targets: DeletionTarget[]) => {
      if (targets.length === 0 || dialogStateRef.current.status === 'trashing' || projectId !== store.project.id) return
      const generation = ++generationRef.current
      setDialogState({ status: 'analyzing', targets })

      try {
        const { plan } = await buildPlan(targets)
        if (generation !== generationRef.current || projectIdRef.current !== projectId) return

        if (!plan.ok) {
          setDialogState({
            status: 'preflight-error',
            targets,
            error: plan.error,
          })
          return
        }

        const eligiblePaths = plan.eligiblePaths
        setDialogState({
          status: 'confirm',
          targets,
          eligiblePaths,
          canTrash: eligiblePaths.length > 0,
        })
      } catch (error) {
        if (generation !== generationRef.current) return
        setDialogState({
          status: 'preflight-error',
          targets,
          error: error instanceof Error ? error.message : String(error),
        })
      }
    },
    [buildPlan, projectId, store],
  )

  const confirmProjectOnly = useCallback(async () => {
    const targets =
      dialogState.status === 'confirm' || dialogState.status === 'preflight-error'
        ? dialogState.targets
        : null
    const latestStore = storeRef.current
    if (!targets || targets.length === 0 || projectIdRef.current !== latestStore.project.id) return

    try {
      latestStore.removeTargets(targets)
      setDialogState({ status: 'idle' })
    } catch (error) {
      setDialogState({ status: 'preflight-error', targets, error: error instanceof Error ? error.message : String(error) })
    }
  }, [dialogState])

  const confirmTrash = useCallback(async () => {
    if (dialogState.status !== 'confirm' || dialogState.eligiblePaths.length === 0) {
      return
    }
    const { targets, eligiblePaths } = dialogState
    const generation = ++generationRef.current

    try {
      const preflight = await buildPlan(targets)
      const { plan, store: latestStore, projectId: latestProjectId } = preflight
      if (generation !== generationRef.current) return
      if (!plan.ok) {
        setDialogState({ status: 'preflight-error', targets, error: plan.error })
        return
      }
      if (plan.eligiblePaths.length === 0 || plan.eligiblePaths.join('\n') !== eligiblePaths.join('\n')) {
        setDialogState({ status: 'confirm', targets, eligiblePaths: plan.eligiblePaths, canTrash: plan.eligiblePaths.length > 0 })
        return
      }
      setDialogState({ status: 'trashing', targets, eligiblePaths: plan.eligiblePaths })
      const result = await window.electronAPI.trashManagedProjectFiles({
        projectId: latestProjectId,
        filePaths: plan.eligiblePaths,
      })
      if (generation !== generationRef.current) return

      if (!result.success) {
        setDialogState({
          status: 'preflight-error',
          targets,
          error: result.error,
        })
        return
      }

      try {
        // Use the exact snapshot validated immediately before IPC; never mutate both stores.
        latestStore.removeTargets(targets)
        if (latestStore.kind === 'editor') latestStore.clearHistory()
      } catch (error) {
        setDialogState({
          status: 'post-trash-error', trashedPaths: result.trashedPaths, failedPaths: result.failedPaths,
          error: error instanceof Error ? error.message : String(error),
        })
        return
      }

      if (result.failedPaths.length === 0) {
        setDialogState({
          status: 'trash-complete',
          trashedPaths: result.trashedPaths,
          failedPaths: [],
        })
      } else {
        setDialogState({
          status: 'trash-partial',
          trashedPaths: result.trashedPaths,
          failedPaths: result.failedPaths,
        })
      }
    } catch (error) {
      if (generation !== generationRef.current) return
      setDialogState({
        status: 'preflight-error',
        targets,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }, [buildPlan, dialogState])

  const cancelDeletion = useCallback(() => {
    if (dialogState.status === 'trashing') return
    generationRef.current += 1
    setDialogState({ status: 'idle' })
  }, [dialogState.status])

  return {
    dialogState,
    requestDeletion,
    confirmProjectOnly,
    confirmTrash,
    cancelDeletion,
  }
}
