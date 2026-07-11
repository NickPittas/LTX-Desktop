import { dialog } from 'electron'
import { randomUUID } from 'crypto'
import os from 'os'
import path from 'path'
import fs from 'fs'
import { getAllowedRoots } from '../config'
import { logger } from '../logger'
import { getMainWindow } from '../window'
import { validatePath, approvePath } from '../path-validation'
import { getProjectAssetsPath, setProjectAssetsPath } from '../app-state'
import { extractVideoFrameToFile, findFfmpegPath, getMediaDurationSeconds, getVideoDimensions, runFfmpeg } from '../export/ffmpeg-utils'
import { createDownsampledThumbnail, getImageDimensions, getThumbnailPaths } from './image-utils'
import { trashManagedProjectFiles } from './managed-file-trash'
import { handle } from './typed-handle'

/** A labeled phase of an asset-import job with a relative weight (0..N). */
interface ImportPhase {
  label: string
  weight: number
  /** When true, the phase reports no determinate fraction (e.g. a single-file copy). */
  indeterminate?: boolean
}

/**
 * Streams overall job progress (0..100, blended across weighted phases) to the
 * renderer via the `asset:importProgress` IPC channel. Determinate phases
 * interpolate their fraction into the overall percent; indeterminate phases
 * emit a single marker at the phase boundary so the toast can show a pulsing
 * bar without a known percent.
 */
function createImportJobTracker(jobId: string, phases: ImportPhase[]) {
  const totalWeight = phases.reduce((sum, p) => sum + p.weight, 0) || 1
  let phaseIdx = -1
  let weightBeforeCurrent = 0

  const emit = (e: { percent?: number; label: string; done?: boolean; indeterminate?: boolean }): void => {
    getMainWindow()?.webContents.send('asset:importProgress', {
      jobId,
      percent: e.percent ?? 0,
      label: e.label,
      done: e.done,
      indeterminate: e.indeterminate,
    })
  }

  return {
    startNext(): void {
      phaseIdx += 1
      const phase = phases[phaseIdx]
      if (!phase) return
      const pct = (weightBeforeCurrent / totalWeight) * 100
      emit({ percent: pct, label: phase.label, indeterminate: phase.indeterminate === true })
    },
    /** Report a determinate fraction (0..1) within the current phase. No-op for indeterminate phases. */
    report(fraction: number): void {
      const phase = phases[phaseIdx]
      if (!phase || phase.indeterminate) return
      const clamped = Math.max(0, Math.min(1, fraction))
      const base = weightBeforeCurrent - phase.weight
      const pct = ((base + phase.weight * clamped) / totalWeight) * 100
      emit({ percent: pct, label: phase.label })
    },
    done(): void {
      emit({ percent: 100, label: 'Done', done: true })
    },
  }
}

const MIME_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.ogg': 'audio/ogg',
  '.aac': 'audio/aac',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
}

function readLocalFileAsBase64(filePath: string): { data: string; mimeType: string } {
  const data = fs.readFileSync(filePath)
  const base64 = data.toString('base64')
  const ext = path.extname(filePath).toLowerCase()
  const mimeType = MIME_TYPES[ext] || 'application/octet-stream'
  return { data: base64, mimeType }
}

function searchDirectoryForFilesImpl(dir: string, filenames: string[]): Record<string, string> {
  const results: Record<string, string> = {}
  const remaining = new Set(filenames.map(f => f.toLowerCase()))

  const walk = (currentDir: string, depth: number) => {
    if (remaining.size === 0 || depth > 10) return
    try {
      const entries = fs.readdirSync(currentDir, { withFileTypes: true })
      for (const entry of entries) {
        if (remaining.size === 0) break
        const fullPath = path.join(currentDir, entry.name)
        if (entry.isFile()) {
          const lower = entry.name.toLowerCase()
          if (remaining.has(lower)) {
            results[lower] = fullPath
            remaining.delete(lower)
          }
        } else if (entry.isDirectory() && !entry.name.startsWith('.')) {
          walk(fullPath, depth + 1)
        }
      }
    } catch {
      // Skip directories we can't read (permissions, etc.)
    }
  }

  walk(dir, 0)
  return results
}

function resolveLocalSourcePath(srcPath: string): string {
  if (!srcPath || !srcPath.trim()) {
    throw new Error('Source path is empty')
  }

  const normalized = srcPath.trim()

  if (!path.isAbsolute(normalized)) {
    throw new Error(`Source path must be absolute: ${srcPath}`)
  }

  const resolved = path.resolve(normalized)
  if (!fs.existsSync(resolved)) {
    throw new Error(`Source path does not exist: ${resolved}`)
  }
  return resolved
}

function getUniqueDestinationPath(destDir: string, fileName: string): string {
  const parsed = path.parse(fileName)
  let candidate = path.join(destDir, fileName)
  let idx = 1
  while (fs.existsSync(candidate)) {
    candidate = path.join(destDir, `${parsed.name}(${idx})${parsed.ext}`)
    idx += 1
  }
  return candidate
}

/** Recursively list all files under `rootDir`, relative to it. */
function listFilesRecursive(rootDir: string): string[] {
  const out: string[] = []
  const walk = (dir: string) => {
    let entries: fs.Dirent[]
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        walk(full)
      } else if (entry.isFile()) {
        out.push(path.relative(rootDir, full))
      }
    }
  }
  walk(rootDir)
  return out
}

function copyToProjectAssetDirectory(srcPath: string, projectId: string): string {
  const assetsRoot = getProjectAssetsPath()
  const destDir = path.join(assetsRoot, projectId)
  fs.mkdirSync(destDir, { recursive: true })
  const fileName = path.basename(srcPath)
  const destPath = getUniqueDestinationPath(destDir, fileName)
  if (fs.statSync(srcPath).isDirectory()) {
    // EXR sequence directory — recursive copy preserving frame filenames.
    fs.cpSync(srcPath, destPath, { recursive: true })
  } else {
    fs.copyFileSync(srcPath, destPath)
  }
  return destPath
}

/**
 * Copy a visual asset into project storage with optional progress reporting.
 *
 * - Directory (EXR sequence): count files first, then copy per-file, reporting
 *   `copied/total` as a 0..1 fraction (determinate).
 * - Single file: report `-1` (indeterminate) before the copy and `1` after, so
 *   the caller can show a pulsing bar while preserving the fast `copyFileSync`
 *   path (sendfile/copy_file_range) — no read/write loop overhead.
 */
function copyVisualAssetWithProgress(
  srcPath: string,
  projectId: string,
  onProgress?: (fraction: number) => void,
): string {
  const assetsRoot = getProjectAssetsPath()
  const destDir = path.join(assetsRoot, projectId)
  fs.mkdirSync(destDir, { recursive: true })
  const fileName = path.basename(srcPath)
  const destPath = getUniqueDestinationPath(destDir, fileName)

  if (fs.statSync(srcPath).isDirectory()) {
    const files = listFilesRecursive(srcPath)
    const total = files.length || 1
    let copied = 0
    for (const rel of files) {
      const srcFile = path.join(srcPath, rel)
      const destFile = path.join(destPath, rel)
      fs.mkdirSync(path.dirname(destFile), { recursive: true })
      fs.copyFileSync(srcFile, destFile)
      copied += 1
      onProgress?.(copied / total)
    }
    onProgress?.(1)
  } else {
    // Single file → indeterminate (keep the fast sendfile-backed copy path).
    onProgress?.(-1)
    fs.copyFileSync(srcPath, destPath)
    onProgress?.(1)
  }
  return destPath
}

function createVideoBigThumbnail(videoPath: string, bigThumbnailPath: string): void {
  extractVideoFrameToFile({
    videoPath,
    seekTime: 0,
    outputPath: bigThumbnailPath,
    timeoutMs: 30000,
  })
}

/**
 * Create a browser-playable sidecar for a project-local primary video.
 *
 * The primary is never touched. ffmpeg writes to an operation-owned temporary
 * file, which is copied exclusively to a collision-safe persistent proxy path.
 */
async function createPersistentVideoProxy(primaryPath: string, onProgress?: (pct: number) => void): Promise<string> {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) {
    throw new Error('ffmpeg not found for video transcoding')
  }

  const { dir, name } = path.parse(primaryPath)
  let tmpPath: string
  while (true) {
    tmpPath = path.join(dir, `${name}_proxy_${randomUUID()}.tmp.mp4`)
    try {
      const tempFile = fs.openSync(tmpPath, 'wx')
      fs.closeSync(tempFile)
      break
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
    }
  }

  const args = [
    '-y',
    '-i', primaryPath,
    '-map', '0:v:0',
    '-map', '0:a?',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'veryfast',
    '-crf', '18',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-movflags', '+faststart',
    tmpPath,
  ]

  // Probe input duration so `out_time_us` can be converted into a 0..1 fraction.
  const durationSec = getMediaDurationSeconds(primaryPath)
  const durationUs = durationSec != null && durationSec > 0
    ? Math.round(durationSec * 1_000_000)
    : undefined

  try {
    // Isolated: import transcodes must NOT be killable by the global export-cancel.
    const result = await runFfmpeg(ffmpegPath, args, { onProgress, isolated: true, durationUs })
    if (!result.success) {
      throw new Error(`Video transcoding failed for ${primaryPath}: ${result.error}`)
    }

    while (true) {
      const proxyPath = path.join(dir, `${name}_proxy_${randomUUID()}.mp4`)
      try {
        fs.copyFileSync(tmpPath, proxyPath, fs.constants.COPYFILE_EXCL)
        return proxyPath
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
      }
    }
  } finally {
    try { fs.unlinkSync(tmpPath) } catch { /* remove only this operation's temporary proxy */ }
  }
}

/**
 * Transcode a user-selected source video to a browser-playable H.264/AAC MP4
 * for preview ONLY. The source is NEVER
 * mutated: nothing is deleted/renamed/overwritten. The preview MP4 is written
 * to a unique file under the OS temp dir and its path is returned. Progress
 * (0..1) is derived from the input's probed duration and forwarded to
 * `onProgress` so the caller can blend it into `asset:importProgress`.
 */
async function transcodeVideoForPreviewImpl(
  srcPath: string,
  onProgress?: (pct: number) => void,
): Promise<string> {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) {
    throw new Error('ffmpeg not found for preview transcoding')
  }

  const outDir = path.join(os.tmpdir(), 'ltx-preview')
  fs.mkdirSync(outDir, { recursive: true })
  const previewPath = path.join(
    outDir,
    `preview_${Date.now()}_${randomUUID()}.mp4`,
  )

  const args = [
    '-y',
    '-i', srcPath,
    '-map', '0:v:0',
    '-map', '0:a?',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'veryfast',
    '-crf', '18',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-movflags', '+faststart',
    previewPath,
  ]

  // Probe input duration so `out_time_us` can be converted into a 0..1 fraction.
  const durationSec = getMediaDurationSeconds(srcPath)
  const durationUs = durationSec != null && durationSec > 0
    ? Math.round(durationSec * 1_000_000)
    : undefined

  // Isolated: preview transcodes must NOT be killable by the global export-cancel
  // and must NOT touch the source file.
  const result = await runFfmpeg(ffmpegPath, args, { onProgress, isolated: true, durationUs })
  if (!result.success) {
    try { fs.existsSync(previewPath) && fs.unlinkSync(previewPath) } catch { /* best-effort cleanup */ }
    throw new Error(`Preview transcoding failed for ${srcPath}: ${result.error}`)
  }
  return previewPath
}

/**
 * Transcode an EXR file or EXR sequence directory to a browser-playable H.264
 * MP4 for preview. EXR sequences are globbed at `-framerate 8`; a single EXR is
 * looped for 1 second. Source is never mutated.
 */
async function transcodeExrForPreviewImpl(
  srcPath: string,
  onProgress?: (pct: number) => void,
): Promise<string> {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) {
    throw new Error('ffmpeg not found for EXR preview transcoding')
  }

  const outDir = path.join(os.tmpdir(), 'ltx-preview')
  fs.mkdirSync(outDir, { recursive: true })
  const previewPath = path.join(outDir, `preview_${Date.now()}_${randomUUID()}.mp4`)

  const stat = fs.statSync(srcPath)
  const isDir = stat.isDirectory()

  let args: string[]
  if (isDir) {
    const exrFiles = fs.readdirSync(srcPath)
      .filter(f => f.toLowerCase().endsWith('.exr'))
      .sort()
    if (exrFiles.length === 0) {
      throw new Error(`No .exr files found in directory: ${srcPath}`)
    }
    // Use glob pattern for the directory input
    const globPattern = path.join(srcPath, '*.exr')
    args = [
      '-y',
      '-framerate', '8',
      '-pattern_type', 'glob',
      '-i', globPattern,
      '-map', '0:v:0',
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
      '-preset', 'veryfast',
      '-crf', '20',
      '-movflags', '+faststart',
      previewPath,
    ]
  } else {
    // Single EXR file: loop for 1 second to make a playable clip
    args = [
      '-y',
      '-loop', '1', '-t', '1',
      '-i', srcPath,
      '-map', '0:v:0',
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
      '-preset', 'veryfast',
      '-crf', '20',
      '-movflags', '+faststart',
      previewPath,
    ]
  }

  // EXR duration is indeterminate; use a synthetic 30-frame estimate for progress.
  const syntheticDurationUs = isDir ? Math.round(30 * 125_000) : 1_000_000
  const result = await runFfmpeg(ffmpegPath, args, { onProgress, isolated: true, durationUs: syntheticDurationUs })
  if (!result.success) {
    try { fs.existsSync(previewPath) && fs.unlinkSync(previewPath) } catch { /* best-effort cleanup */ }
    throw new Error(`EXR preview transcoding failed for ${srcPath}: ${result.error}`)
  }
  return previewPath
}

// Every video primary is preserved verbatim. A supplied proxy is copied
// alongside; otherwise a persistent H.264/AAC sidecar is generated locally.

function createVisualThumbnails(assetPath: string, type: 'video' | 'image'): { bigThumbnailPath: string; smallThumbnailPath: string } {
  const { bigThumbnailPath: generatedBigThumbnailPath, smallThumbnailPath } = getThumbnailPaths(assetPath)
  let bigThumbnailPath: string

  switch (type) {
    case 'video':
      bigThumbnailPath = generatedBigThumbnailPath
      createVideoBigThumbnail(assetPath, bigThumbnailPath)
      break
    case 'image':
      bigThumbnailPath = assetPath
      break
    default: {
      const unsupportedType: never = type
      throw new Error(`Unsupported visual asset type: ${unsupportedType}`)
    }
  }

  createDownsampledThumbnail(bigThumbnailPath, smallThumbnailPath)
  return { bigThumbnailPath, smallThumbnailPath }
}

function getVisualAssetDimensions(assetPath: string, type: 'video' | 'image'): { width: number; height: number } {
  switch (type) {
    case 'video':
      return getVideoDimensions(assetPath)
    case 'image':
      return getImageDimensions(assetPath)
    default: {
      const unsupportedType: never = type
      throw new Error(`Unsupported visual asset type: ${unsupportedType}`)
    }
  }
}

export function registerFileHandlers(): void {
  handle('openLtxApiKeyPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://console.ltx.video/api-keys/')
    return true
  })

  handle('openLtxBillingPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://console.ltx.video/billings/#buy')
    return true
  })

  handle('openFalApiKeyPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://fal.ai/dashboard/keys')
    return true
  })

  handle('openHuggingFaceRepo', async ({ repoId }) => {
    const { shell } = await import('electron')
    await shell.openExternal(`https://huggingface.co/${repoId}`)
    return true
  })

  const HF_AUTHORIZE_URL = 'https://huggingface.co/oauth/authorize'

  handle('openHuggingFaceAuth', async (params) => {
    const { shell } = await import('electron')
    const url = new URL(HF_AUTHORIZE_URL)
    url.searchParams.set('client_id', params.clientId)
    url.searchParams.set('redirect_uri', params.redirectUri)
    url.searchParams.set('response_type', 'code')
    url.searchParams.set('scope', params.scope)
    url.searchParams.set('state', params.state)
    url.searchParams.set('code_challenge', params.codeChallenge)
    url.searchParams.set('code_challenge_method', params.codeChallengeMethod)
    await shell.openExternal(url.toString())
    return true
  })

  handle('openParentFolderOfFile', async ({ filePath }) => {
    const { shell } = await import('electron')
    const normalizedPath = validatePath(filePath, getAllowedRoots())
    const parentDir = path.dirname(normalizedPath)
    if (!fs.existsSync(parentDir) || !fs.statSync(parentDir).isDirectory()) {
      throw new Error(`Parent directory not found: ${parentDir}`)
    }
    shell.openPath(parentDir)
  })

  handle('showItemInFolder', async ({ filePath }) => {
    const { shell } = await import('electron')
    shell.showItemInFolder(filePath)
  })

  handle('readLocalFile', async ({ filePath }) => {
    try {
      const normalizedPath = validatePath(filePath, getAllowedRoots())

      if (!fs.existsSync(normalizedPath)) {
        throw new Error(`File not found: ${normalizedPath}`)
      }

      return readLocalFileAsBase64(normalizedPath)
    } catch (error) {
      logger.error( `Error reading local file: ${error}`)
      throw error
    }
  })

  handle('showSaveDialog', async ({ title, defaultPath, filters }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showSaveDialog(mainWindow, {
      title: title || 'Save File',
      defaultPath,
      filters: filters || [],
    })
    if (result.canceled || !result.filePath) return null
    approvePath(result.filePath)
    return result.filePath
  })

  handle('saveFile', async ({ filePath, data, encoding }) => {
    try {
      validatePath(filePath, getAllowedRoots())
      if (encoding === 'base64') {
        fs.writeFileSync(filePath, Buffer.from(data, 'base64'))
      } else {
        fs.writeFileSync(filePath, data, 'utf-8')
      }
      return { success: true, path: filePath }
    } catch (error) {
      logger.error( `Error saving file: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('saveBinaryFile', async ({ filePath, data }) => {
    try {
      validatePath(filePath, getAllowedRoots())
      fs.writeFileSync(filePath, Buffer.from(data))
      return { success: true, path: filePath }
    } catch (error) {
      logger.error( `Error saving binary file: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('showOpenDirectoryDialog', async ({ title }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select Folder',
      properties: ['openDirectory', 'createDirectory'],
    })
    if (result.canceled || result.filePaths.length === 0) return null
    approvePath(result.filePaths[0])
    return result.filePaths[0]
  })

  handle('searchDirectoryForFiles', ({ directory, filenames }) => {
    return searchDirectoryForFilesImpl(directory, filenames)
  })

  handle('addVisualAssetToProject', async ({ srcPath, projectId, type, proxyPath, jobId }) => {
    const id = jobId && jobId.length > 0 ? jobId : randomUUID()

    const fail = (error: unknown): { success: false; error: string } => {
      // Ensure the toast dismisses even when we throw before tracker.done().
      getMainWindow()?.webContents.send('asset:importProgress', {
        jobId: id,
        percent: 0,
        label: 'Failed',
        done: true,
      })
      logger.error(`Error adding asset to project: ${error}`)
      return { success: false, error: String(error) }
    }

    try {
      const resolvedSrc = resolveLocalSourcePath(srcPath)
      const srcIsDir = fs.statSync(resolvedSrc).isDirectory()

      // Build the phase plan for this import path so overall progress maps
      // determinate phases (dir copy, transcode) into weighted sub-ranges and
      // marks quick/unknowable phases (single-file copy, finalize) indeterminate.
      const phases: ImportPhase[] = []
      if (proxyPath) {
        // ProRes/EXR primary preserved verbatim + proxy copied alongside.
        // No transcode (Phase 3a primary-preservation).
        phases.push({ label: 'Importing asset…', weight: 40, indeterminate: !srcIsDir })
        phases.push({ label: 'Importing preview…', weight: 50, indeterminate: true })
        phases.push({ label: 'Finalizing…', weight: 10, indeterminate: true })
      } else if (type === 'video') {
        // No-proxy path: primary copied unchanged + persistent H.264 proxy sidecar generated.
        phases.push({ label: 'Importing asset…', weight: 5, indeterminate: !srcIsDir })
        phases.push({ label: 'Transcoding preview…', weight: 90 })
        phases.push({ label: 'Finalizing…', weight: 5, indeterminate: true })
      } else {
        phases.push({ label: 'Importing asset…', weight: 80, indeterminate: !srcIsDir })
        phases.push({ label: 'Finalizing…', weight: 20, indeterminate: true })
      }

      const tracker = createImportJobTracker(id, phases)

      // Phase 1: copy primary
      tracker.startNext()
      const destPath = copyVisualAssetWithProgress(resolvedSrc, projectId, (fraction) => {
        // fraction < 0 === indeterminate marker; ignore (already emitted at start).
        if (fraction >= 0) tracker.report(fraction)
      })

      let projectProxyPath: string | undefined
      let thumbnailSourcePath = destPath

      if (proxyPath) {
        // Phase 2 (proxy path): copy proxy MP4 alongside (no transcode —
        // primary preserved verbatim per Phase 3a).
        tracker.startNext()
        const resolvedProxy = resolveLocalSourcePath(proxyPath)
        projectProxyPath = copyVisualAssetWithProgress(resolvedProxy, projectId)
        thumbnailSourcePath = projectProxyPath
      } else if (type === 'video') {
        // Phase 2 (no-proxy video path): generate a persistent H.264 proxy sidecar.
        tracker.startNext()
        projectProxyPath = await createPersistentVideoProxy(destPath, (pct) => tracker.report(pct))
        thumbnailSourcePath = projectProxyPath
      }

      // Final phase: thumbnails + dimensions from the browser-playable source.
      tracker.startNext()
      const { bigThumbnailPath, smallThumbnailPath } = createVisualThumbnails(thumbnailSourcePath, type)
      const { width, height } = getVisualAssetDimensions(thumbnailSourcePath, type)

      tracker.done()

      return {
        success: true,
        path: destPath,
        proxyPath: projectProxyPath,
        bigThumbnailPath,
        smallThumbnailPath,
        width,
        height,
      }
    } catch (error) {
      return fail(error)
    }
  })

  handle('addGenericAssetToProject', ({ srcPath, projectId }) => {
    try {
      const resolvedSrc = resolveLocalSourcePath(srcPath)
      const destPath = copyToProjectAssetDirectory(resolvedSrc, projectId)
      return { success: true, path: destPath }
    } catch (error) {
      logger.error(`Error copying file to project assets: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('trashManagedProjectFiles', async ({ projectId, filePaths }) => {
    const { shell } = await import('electron')
    const projectAssetsRoot = getProjectAssetsPath()
    const generationRoot = path.join(projectAssetsRoot, '.ltx-generations')
    return trashManagedProjectFiles(
      { projectId, filePaths },
      { projectAssetsRoot, generationRoot, trashItem: shell.trashItem },
    )
  })

  handle('makeThumbnailsForProjectAsset', ({ path: assetPath, type }) => {
    try {
      const resolvedAssetPath = resolveLocalSourcePath(assetPath)
      const { bigThumbnailPath, smallThumbnailPath } = createVisualThumbnails(resolvedAssetPath, type)

      return {
        success: true,
        bigThumbnailPath,
        smallThumbnailPath,
      }
    } catch (error) {
      logger.error(`Error creating thumbnails for project asset: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('makeDimensionsForProjectAsset', ({ path: assetPath, type }) => {
    try {
      const resolvedAssetPath = resolveLocalSourcePath(assetPath)
      const { width, height } = getVisualAssetDimensions(resolvedAssetPath, type)

      return {
        success: true,
        width,
        height,
      }
    } catch (error) {
      logger.error(`Error creating dimensions for project asset: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('transcodeVideoForPreview', async ({ srcPath, jobId }) => {
    const id = jobId && jobId.length > 0 ? jobId : randomUUID()

    const emit = (e: { percent?: number; label: string; done?: boolean }): void => {
      getMainWindow()?.webContents.send('asset:importProgress', {
        jobId: id,
        percent: e.percent ?? 0,
        label: e.label,
        done: e.done,
      })
    }

    try {
      const resolvedSrc = resolveLocalSourcePath(srcPath)
      const srcStat = fs.statSync(resolvedSrc)
      const isExrDir = srcStat.isDirectory()
      const isExrFile = !srcStat.isDirectory() && resolvedSrc.toLowerCase().endsWith('.exr')

      if (isExrDir || isExrFile) {
        emit({ percent: 0, label: 'Transcoding EXR preview…' })
        const previewPath = await transcodeExrForPreviewImpl(resolvedSrc, (pct) => {
          emit({ percent: Math.round(pct * 100), label: 'Transcoding EXR preview…' })
        })
        emit({ percent: 100, label: 'Preview ready', done: true })
        return { success: true, path: previewPath }
      }

      if (srcStat.isDirectory()) {
        throw new Error('Preview transcoding is only supported for video/EXR files, not directories')
      }

      emit({ percent: 0, label: 'Transcoding preview…' })
      const previewPath = await transcodeVideoForPreviewImpl(resolvedSrc, (pct) => {
        emit({ percent: Math.round(pct * 100), label: 'Transcoding preview…' })
      })
      emit({ percent: 100, label: 'Preview ready', done: true })

      return { success: true, path: previewPath }
    } catch (error) {
      logger.error(`Error transcoding preview video: ${error}`)
      // Terminal event so any listener toast dismisses on failure.
      emit({ percent: 0, label: 'Failed', done: true })
      return { success: false, error: String(error) }
    }
  })

  handle('getProjectAssetsPath', () => {
    return getProjectAssetsPath()
  })

  handle('openProjectAssetsPathChangeDialog', async () => {
    try {
      const mainWindow = getMainWindow()
      if (!mainWindow) return { success: false, error: 'No window' }
      const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Project Assets Path',
        properties: ['openDirectory', 'createDirectory'],
      })
      if (result.canceled || result.filePaths.length === 0) return { success: false, error: 'cancelled' }
      const selectedPath = path.resolve(result.filePaths[0])
      setProjectAssetsPath(selectedPath)
      approvePath(selectedPath)
      return { success: true, path: selectedPath }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  })

  handle('checkFilesExist', ({ filePaths }) => {
    const results: Record<string, boolean> = {}
    for (const p of filePaths) {
      try {
        results[p] = fs.existsSync(p)
      } catch {
        results[p] = false
      }
    }
    return results
  })

  handle('showOpenFileDialog', async ({ title, filters, properties }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const props: any[] = ['openFile']
    if (properties?.includes('multiSelections')) props.push('multiSelections')
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select File',
      filters: filters || [],
      properties: props,
    })
    if (result.canceled || result.filePaths.length === 0) return null
    for (const fp of result.filePaths) {
      approvePath(fp)
    }
    return result.filePaths
  })

}
