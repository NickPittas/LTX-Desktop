import { logger } from './logger'

export type ProjectAssetType = 'video' | 'image'

export interface ProjectAssetCopyResult {
  path: string
  proxyPath?: string | null
  bigThumbnailPath: string
  smallThumbnailPath: string
  width: number
  height: number
}

/**
 * Copy a video/image file to project storage and return precomputed thumbnail paths.
 * When ``proxyPath`` is supplied (ProRes/EXR primary), the primary is preserved
 * verbatim (no transcode) and the proxy is copied alongside — thumbnails are
 * generated from the proxy.
 */
export async function addVisualAssetToProject(
  srcPath: string,
  projectId: string,
  type: ProjectAssetType,
  proxyPath?: string,
): Promise<ProjectAssetCopyResult | null> {
  try {
    const result = await window.electronAPI.addVisualAssetToProject({ srcPath, projectId, type, proxyPath })
    if (result.success) {
      return {
        path: result.path,
        proxyPath: result.proxyPath,
        bigThumbnailPath: result.bigThumbnailPath,
        smallThumbnailPath: result.smallThumbnailPath,
        width: result.width,
        height: result.height,
      }
    }
    logger.warn(`Failed to add asset to project folder: ${result.error}`)
  } catch (e) {
    logger.warn(`Failed to add asset to project folder: ${e}`)
  }
  return null
}

/**
 * Copy a file to project storage without thumbnail generation (audio path).
 */
export async function addGenericAssetToProject(
  srcPath: string,
  projectId: string,
): Promise<{ path: string } | null> {
  try {
    const result = await window.electronAPI.addGenericAssetToProject({ srcPath, projectId })
    if (result.success) {
      return { path: result.path }
    }
    logger.warn(`Failed to copy file to project folder: ${result.error}`)
  } catch (e) {
    logger.warn(`Failed to copy file to project folder: ${e}`)
  }
  return null
}
