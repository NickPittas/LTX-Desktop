import type { Asset, AssetTake, Project } from '../types/project-model'
import type { EditorModel } from '../views/editor/editor-state'

export interface DeletionTarget {
  assetId: string
  takeIndex?: number
}

export type ProjectReferenceReadResult =
  | { ok: true; projects: Project[] }
  | { ok: false; error: string }

export type ManagedDeletionPlan =
  | { ok: true; eligiblePaths: string[] }
  | { ok: false; error: string }

type PathKind = 'project-copy' | 'source'

interface CanonicalRoot {
  value: string
  windows: boolean
}

interface TargetedAsset {
  asset: Asset
  takeIndexes: Set<number> | null
}

function canonicalizeAbsolutePath(value: string, root: string): string | null {
  if (!value || value.includes('\0')) return null
  const windows = /^[A-Za-z]:[\\/]/.test(root)
  const absolute = windows ? /^[A-Za-z]:[\\/]/.test(value) : value.startsWith('/')
  if (!absolute || (windows && value.slice(0, 2).toLowerCase() !== root.slice(0, 2).toLowerCase())) return null

  const separator = windows ? '\\' : '/'
  const normalized = value.replace(/[\\/]+/g, separator)
  const prefix = windows ? normalized.slice(0, 2).toLowerCase() : ''
  const parts = normalized.slice(windows ? 2 : 1).split(separator)
  const resolved: string[] = []
  for (const part of parts) {
    if (!part || part === '.') continue
    if (part === '..') {
      if (resolved.length === 0) return null
      resolved.pop()
    } else {
      resolved.push(part)
    }
  }
  return windows ? `${prefix}${separator}${resolved.join(separator)}` : `${separator}${resolved.join(separator)}`
}

function canonicalRoot(value: string | null): CanonicalRoot | null {
  if (!value) return null
  const normalized = canonicalizeAbsolutePath(value, value)
  if (!normalized) return null
  return { value: normalized, windows: /^[A-Za-z]:[\\/]/.test(normalized) }
}

function isWithinRoot(path: string, root: CanonicalRoot): boolean {
  const comparablePath = root.windows ? path.toLowerCase() : path
  const comparableRoot = root.windows ? root.value.toLowerCase() : root.value
  const separator = root.windows ? '\\' : '/'
  return comparablePath === comparableRoot || comparablePath.startsWith(
    comparableRoot.endsWith(separator) ? comparableRoot : `${comparableRoot}${separator}`,
  )
}

function canonicalizeManagedPath(value: string | undefined, kind: PathKind, roots: { project: CanonicalRoot; generation: CanonicalRoot | null }): string | null {
  if (typeof value !== 'string') return null
  const root = kind === 'project-copy' ? roots.project : roots.generation
  if (!root) return null
  const path = canonicalizeAbsolutePath(value, root.value)
  return path && isWithinRoot(path, root) ? path : null
}

function assetPaths(asset: Asset | AssetTake, roots: { project: CanonicalRoot; generation: CanonicalRoot | null }): string[] {
  const paths = [
    canonicalizeManagedPath(asset.path, 'project-copy', roots),
    canonicalizeManagedPath(asset.proxyPath, 'project-copy', roots),
    canonicalizeManagedPath(asset.bigThumbnailPath, 'project-copy', roots),
    canonicalizeManagedPath(asset.smallThumbnailPath, 'project-copy', roots),
    ...(asset.managedSourcePaths ?? []).map(path => canonicalizeManagedPath(path, 'source', roots)),
  ]
  return paths.filter((path): path is string => path !== null)
}

function projectWithEditorModel(project: Project, editorModel: EditorModel): Project {
  return {
    ...project,
    assets: editorModel.assets,
    bins: editorModel.bins,
    timelines: editorModel.timelines,
    activeTimelineId: editorModel.activeTimelineId ?? editorModel.timelines[0]?.id,
  }
}

function targetAssets(project: Project, targets: DeletionTarget[]): TargetedAsset[] {
  const allAssetIds = new Set<string>()
  const takeIndexes = new Map<string, Set<number>>()
  for (const target of targets) {
    if (!target.assetId) throw new Error('Deletion target assetId is required')
    if (target.takeIndex === undefined) {
      allAssetIds.add(target.assetId)
      takeIndexes.delete(target.assetId)
      continue
    }
    if (!Number.isInteger(target.takeIndex) || target.takeIndex < 0) throw new Error('Deletion target takeIndex is invalid')
    if (!allAssetIds.has(target.assetId)) {
      let indexes = takeIndexes.get(target.assetId)
      if (!indexes) {
        indexes = new Set()
        takeIndexes.set(target.assetId, indexes)
      }
      indexes.add(target.takeIndex)
    }
  }

  const matched: TargetedAsset[] = []
  for (const asset of project.assets) {
    if (allAssetIds.has(asset.id)) {
      matched.push({ asset, takeIndexes: null })
      continue
    }
    const indexes = takeIndexes.get(asset.id)
    if (indexes) matched.push({ asset, takeIndexes: indexes })
  }
  return matched
}

function projectAfterDeletion(project: Project, targets: TargetedAsset[], editorMode: boolean): Project {
  const assetsToDelete = new Set(targets.filter(target => target.takeIndexes === null).map(target => target.asset.id))
  const takeDeletes = new Map(targets.filter(target => target.takeIndexes !== null).map(target => [target.asset.id, target.takeIndexes!]))
  const remainingAssets = project.assets.flatMap(asset => {
    if (assetsToDelete.has(asset.id)) return []
    const deleted = takeDeletes.get(asset.id)
    if (!deleted || !asset.takes || asset.takes.length <= 1) return [asset]
    const validDeletes = new Set([...deleted].filter(index => index < asset.takes!.length))
    if (validDeletes.size === 0 || validDeletes.size >= asset.takes.length) return [asset]

    const takes = asset.takes.filter((_, index) => !validDeletes.has(index))
    const activeTakeIndex = Math.max(0, Math.min(asset.activeTakeIndex ?? (asset.takes.length - 1), takes.length - 1))
    const activeTake = takes[activeTakeIndex]
    return [{
      ...asset,
      takes,
      activeTakeIndex,
      path: activeTake.path,
      origin: activeTake.origin,
      managedSourcePaths: activeTake.managedSourcePaths,
      proxyPath: activeTake.proxyPath,
      bigThumbnailPath: activeTake.bigThumbnailPath,
      smallThumbnailPath: activeTake.smallThumbnailPath,
      width: activeTake.width,
      height: activeTake.height,
    }]
  })

  const repointTakeIndex = (assetId: string, takeIndex: number | undefined): number | undefined => {
    if (takeIndex === undefined) return undefined
    const deleted = takeDeletes.get(assetId)
    const original = project.assets.find(asset => asset.id === assetId)
    if (!deleted || !original?.takes) return takeIndex
    const validDeletes = [...deleted].filter(index => index < original.takes!.length)
    if (validDeletes.length === 0 || validDeletes.length >= original.takes.length) return takeIndex
    if (!validDeletes.includes(takeIndex)) return takeIndex - validDeletes.filter(index => index < takeIndex).length
    const survivors = original.takes.map((_, index) => index).filter(index => !deleted.has(index))
    const preceding = survivors.filter(index => index < takeIndex)
    const replacement = preceding[preceding.length - 1] ?? survivors[0]
    return survivors.indexOf(replacement)
  }

  const timelines = !editorMode ? project.timelines : project.timelines.map(timeline => ({
    ...timeline,
    clips: timeline.clips
      .filter(clip => !clip.assetId || !assetsToDelete.has(clip.assetId))
      .map(clip => ({ ...clip, takeIndex: repointTakeIndex(clip.assetId ?? '', clip.takeIndex) })),
  }))

  return {
    ...project,
    assets: remainingAssets,
    timelines,
  }
}

function referencePaths(project: Project, roots: { project: CanonicalRoot; generation: CanonicalRoot | null }): string[] {
  const projectAssetPaths = project.assets.flatMap(asset => [
    ...assetPaths(asset, roots),
    ...(asset.takes ?? []).flatMap(take => assetPaths(take, roots)),
  ])
  const timelineAssetPaths = project.timelines.flatMap(timeline => timeline.clips.flatMap(clip => clip.asset ? [
      ...assetPaths(clip.asset, roots),
      ...(clip.asset.takes ?? []).flatMap(take => assetPaths(take, roots)),
    ] : []))
  return [...projectAssetPaths, ...timelineAssetPaths]
}

/** Plans only: it never mutates projects or files, and fails closed when storage cannot be read. */
export function buildManagedDeletionPlan({
  persistedProjects,
  currentProject,
  editorSnapshot,
  targets,
  projectManagedDirectory,
  generationRoot,
}: {
  persistedProjects: ProjectReferenceReadResult
  currentProject: Project
  editorSnapshot: { projectId: string; model: EditorModel } | null
  targets: DeletionTarget[]
  projectManagedDirectory: string
  generationRoot: string | null
}): ManagedDeletionPlan {
  if (!persistedProjects.ok) return persistedProjects

  try {
    const projectRoot = canonicalRoot(projectManagedDirectory)
    if (!projectRoot) throw new Error('Project managed directory is invalid')
    if (projectRoot.value === '/' || (projectRoot.windows && /^[a-z]:\\$/i.test(projectRoot.value))) {
      throw new Error('Project managed directory must not be a filesystem root')
    }
    const roots = { project: projectRoot, generation: canonicalRoot(generationRoot) }
    if (generationRoot !== null && !roots.generation) throw new Error('Generation root is invalid')
    if (editorSnapshot && editorSnapshot.projectId !== currentProject.id) {
      throw new Error('Editor snapshot project does not match current project')
    }

    const effectiveCurrentProject = editorSnapshot
      ? projectWithEditorModel(currentProject, editorSnapshot.model)
      : currentProject
    const targetedAssets = targetAssets(effectiveCurrentProject, targets)
    const candidates = new Set<string>()
    for (const target of targetedAssets) {
      if (target.takeIndexes === null) {
        if (target.asset.origin === 'generated') {
          assetPaths(target.asset, roots).forEach(path => candidates.add(path))
        }
        // A generated sibling take remains managed even if the parent currently aliases an imported take.
        for (const take of target.asset.takes ?? []) if (take.origin === 'generated') assetPaths(take, roots).forEach(path => candidates.add(path))
      } else {
        for (const index of target.takeIndexes) {
          const take = target.asset.takes?.[index]
          if (take?.origin === 'generated') assetPaths(take, roots).forEach(path => candidates.add(path))
        }
      }
    }

    const projectedCurrent = projectAfterDeletion(effectiveCurrentProject, targetedAssets, editorSnapshot !== null)
    const references = new Set([
      ...referencePaths(projectedCurrent, roots),
      ...persistedProjects.projects.filter(project => project.id !== currentProject.id).flatMap(project => referencePaths(project, roots)),
    ])
    return { ok: true, eligiblePaths: [...candidates].filter(path => !references.has(path)).sort() }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}
