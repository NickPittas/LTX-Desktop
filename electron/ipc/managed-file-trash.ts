import fs from 'fs'
import path from 'path'

export interface TrashManagedProjectFilesInput {
  projectId: string
  filePaths: string[]
}

export interface TrashManagedProjectFilesDependencies {
  projectAssetsRoot: string
  generationRoot: string
  trashItem: (filePath: string) => Promise<void>
}

export type TrashManagedProjectFilesResult =
  | { success: true; trashedPaths: string[]; failedPaths: Array<{ path: string; reason: string }> }
  | { success: false; error: string }

function isFilesystemRoot(filePath: string): boolean {
  return path.parse(filePath).root === filePath
}

function isDirectChildOf(candidate: string, root: string): boolean {
  const relative = path.relative(root, candidate)
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)
}

function rejectSymlinkPathComponents(root: string, candidate: string): void {
  let component = root
  for (const segment of path.relative(root, candidate).split(path.sep)) {
    if (fs.lstatSync(component).isSymbolicLink()) {
      throw new Error(`Managed file path contains a symbolic link: ${component}`)
    }
    component = path.join(component, segment)
  }
  if (fs.lstatSync(component).isSymbolicLink()) {
    throw new Error(`Managed file path contains a symbolic link: ${component}`)
  }
}

function canonicalDirectory(directory: string, label: string): string {
  if (!path.isAbsolute(directory)) throw new Error(`${label} must be absolute`)
  const resolved = path.resolve(directory)
  if (!fs.statSync(resolved).isDirectory()) throw new Error(`${label} must be a directory`)
  return fs.realpathSync.native(resolved)
}

function optionalCanonicalManagedDirectory(directory: string, expectedCanonicalDirectory: string, label: string): string | undefined {
  try {
    const entry = fs.lstatSync(directory)
    if (entry.isSymbolicLink()) throw new Error(`${label} must not be a symbolic link`)
    if (!entry.isDirectory()) throw new Error(`${label} must be a directory`)
    const canonical = fs.realpathSync.native(directory)
    if (canonical !== expectedCanonicalDirectory) throw new Error(`${label} is not the expected managed directory`)
    return canonical
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
}

function validateProjectId(projectId: string): void {
  if (
    !projectId ||
    projectId !== projectId.trim() ||
    projectId === '.' ||
    projectId === '..' ||
    projectId.startsWith('.') ||
    projectId.includes('/') ||
    projectId.includes('\\') ||
    path.isAbsolute(projectId)
  ) {
    throw new Error('Project ID is unsafe')
  }
}

/**
 * Trashes only files owned by one project or the managed generation root.
 * All filesystem ownership checks finish before the first trash callback.
 */
export async function trashManagedProjectFiles(
  { projectId, filePaths }: TrashManagedProjectFilesInput,
  { projectAssetsRoot, generationRoot, trashItem }: TrashManagedProjectFilesDependencies,
): Promise<TrashManagedProjectFilesResult> {
  try {
    validateProjectId(projectId)
    if (!Array.isArray(filePaths)) throw new Error('File paths must be an array')
    if (filePaths.length === 0) return { success: true, trashedPaths: [], failedPaths: [] }

    const resolvedAssetsRoot = path.resolve(projectAssetsRoot)
    const canonicalAssetsRoot = canonicalDirectory(resolvedAssetsRoot, 'Project assets root')
    if (isFilesystemRoot(canonicalAssetsRoot)) throw new Error('Project assets root must not be a filesystem root')

    const projectDirectory = path.join(resolvedAssetsRoot, projectId)
    const expectedGenerationRoot = path.resolve(generationRoot)
    if (expectedGenerationRoot !== path.join(resolvedAssetsRoot, '.ltx-generations')) {
      throw new Error('Generation root must be the managed generation directory')
    }

    const canonicalProjectDirectory = optionalCanonicalManagedDirectory(
      projectDirectory,
      path.join(canonicalAssetsRoot, projectId),
      'Project directory',
    )
    const canonicalGenerationRoot = optionalCanonicalManagedDirectory(
      expectedGenerationRoot,
      path.join(canonicalAssetsRoot, '.ltx-generations'),
      'Generation root',
    )

    const candidates = new Set<string>()
    for (const filePath of filePaths) {
      if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) {
        throw new Error('Managed file path must be absolute')
      }

      const resolved = path.resolve(filePath)
      if (isFilesystemRoot(resolved)) throw new Error('Managed file path must not be a filesystem root')
      let entry: fs.Stats
      try {
        entry = fs.lstatSync(resolved)
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') throw new Error(`Managed file does not exist: ${resolved}`)
        throw error
      }
      if (entry.isSymbolicLink()) throw new Error(`Managed file must not be a symbolic link: ${resolved}`)
      if (fs.statSync(resolved).isDirectory()) throw new Error(`Managed file must not be a directory: ${resolved}`)

      const lexicalProjectPath = isDirectChildOf(resolved, projectDirectory)
      const lexicalGenerationPath = isDirectChildOf(resolved, expectedGenerationRoot)
      if (!lexicalProjectPath && !lexicalGenerationPath) {
        throw new Error(`Managed file is outside allowed roots: ${resolved}`)
      }

      rejectSymlinkPathComponents(lexicalProjectPath ? projectDirectory : expectedGenerationRoot, resolved)
      const canonical = fs.realpathSync.native(resolved)
      const ownedByProject = lexicalProjectPath && canonicalProjectDirectory && isDirectChildOf(canonical, canonicalProjectDirectory)
      const ownedByGeneration = lexicalGenerationPath && canonicalGenerationRoot && isDirectChildOf(canonical, canonicalGenerationRoot)
      if (!ownedByProject && !ownedByGeneration) {
        throw new Error(`Managed file escapes allowed roots: ${resolved}`)
      }
      candidates.add(canonical)
    }

    const trashedPaths: string[] = []
    const failedPaths: Array<{ path: string; reason: string }> = []
    for (const candidate of Array.from(candidates)) {
      try {
        await trashItem(candidate)
        trashedPaths.push(candidate)
      } catch (error) {
        failedPaths.push({ path: candidate, reason: error instanceof Error ? error.message : String(error) })
      }
    }
    return { success: true, trashedPaths, failedPaths }
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : String(error) }
  }
}
