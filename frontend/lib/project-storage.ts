import { migrateProjectData, projectSchema, type Project } from '../types/project-model'
import type { ProjectReferenceReadResult } from './managed-file-references'
import { logger } from './logger'

export const PROJECT_IDS_STORAGE_KEY = 'ltx-project-ids'
export const PROJECT_STORAGE_KEY_PREFIX = 'ltx-project-'

export function getProjectStorageKey(projectId: string): string {
  return `${PROJECT_STORAGE_KEY_PREFIX}${projectId}`
}

export function readProjectIds(): string[] {
  try {
    const stored = localStorage.getItem(PROJECT_IDS_STORAGE_KEY)
    if (!stored) return []

    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) {
      logger.error('Project ids payload is not an array')
      return []
    }

    return parsed.filter((projectId): projectId is string => typeof projectId === 'string')
  } catch (error) {
    logger.error(`Failed to read project ids: ${error}`)
    return []
  }
}

export function writeProjectIds(projectIds: string[]): void {
  localStorage.setItem(
    PROJECT_IDS_STORAGE_KEY,
    JSON.stringify(Array.from(new Set(projectIds))),
  )
}

export function readProject(projectId: string): Project | null {
  try {
    const stored = localStorage.getItem(getProjectStorageKey(projectId))
    if (!stored) return null

    const { project, migrated } = migrateProjectData(JSON.parse(stored))
    const normalizedProject = project.id === projectId
      ? project
      : projectSchema.parse({ ...project, id: projectId })

    if (migrated || normalizedProject.id !== project.id) {
      writeProject(projectId, normalizedProject)
    }

    return normalizedProject
  } catch (error) {
    logger.error(`Failed to read project ${projectId}: ${error}`)
    return null
  }
}

/** Reads every persisted project for fail-closed managed-file reference analysis. */
export function readAllProjectsForReferenceAnalysis(): ProjectReferenceReadResult {
  try {
    const storedIds = localStorage.getItem(PROJECT_IDS_STORAGE_KEY)
    if (!storedIds) {
      if (localStorage.getItem('ltx-projects') !== null) {
        return { ok: false, error: 'Legacy project migration is pending' }
      }
      return { ok: true, projects: [] }
    }

    const parsedIds: unknown = JSON.parse(storedIds)
    if (!Array.isArray(parsedIds) || !parsedIds.every(projectId => typeof projectId === 'string')) {
      throw new Error('Project ids payload is not an array of strings')
    }

    const projects = Array.from(new Set(parsedIds)).map(projectId => {
      const stored = localStorage.getItem(getProjectStorageKey(projectId))
      if (!stored) throw new Error(`Persisted project ${projectId} is missing`)

      const { project } = migrateProjectData(JSON.parse(stored))
      return project.id === projectId
        ? project
        : projectSchema.parse({ ...project, id: projectId })
    })
    return { ok: true, projects }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

export function writeProject(projectId: string, project: Project): Project {
  const normalizedProject = projectSchema.parse({ ...project, id: projectId })
  localStorage.setItem(
    getProjectStorageKey(projectId),
    JSON.stringify(normalizedProject),
  )
  return normalizedProject
}

export function deleteProjectEntry(projectId: string): void {
  localStorage.removeItem(getProjectStorageKey(projectId))
}
