import fs from 'fs'
import os from 'os'
import path from 'path'
import { getAllowedRoots } from '../config'
import { validatePath } from '../path-validation'
import { findFfmpegPath, runFfmpeg, stopExportProcess } from './ffmpeg-utils'
import { flattenTimeline } from './timeline'
import { mixAudioToPcm } from './audio-mix'
import { buildExportVideoPlan } from './export-codec'
import { handle } from '../ipc/typed-handle'
import { runExportPipeline, type ExportRunnerDependencies } from './export-runner'

const defaultDependencies: ExportRunnerDependencies = {
  run: runFfmpeg, mix: mixAudioToPcm, writeFile: fs.writeFileSync,
  unlink: filePath => { try { fs.unlinkSync(filePath) } catch {} },
  rename: fs.renameSync, tmpDir: os.tmpdir(),
  unique: () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
}

let activeExportController: AbortController | null = null

export function registerExportHandlers(): void {
  handle('exportNative', async (input) => {
    if (activeExportController) return { success: false, error: 'An export is already in progress' }
    const ffmpegPath = findFfmpegPath()
    if (!ffmpegPath) return { success: false, error: 'FFmpeg not found' }
    try {
      buildExportVideoPlan(input.codec, input.quality)
      validatePath(input.outputPath, getAllowedRoots())
      for (const clip of input.clips) {
        if (clip.path) validatePath(clip.path, getAllowedRoots())
        if (clip.audioPath) validatePath(clip.audioPath, getAllowedRoots())
        if ((clip.type === 'audio' || clip.type === 'video') && !clip.muted && clip.volume > 0) {
          const audioSource = clip.audioPath ?? clip.path
          if (!audioSource || !fs.existsSync(audioSource)) return { success: false, error: `Audio source file not found: ${audioSource ?? '(missing)'}` }
        }
      }
      const segments = flattenTimeline(input.clips)
      for (const segment of segments) if (segment.filePath && !fs.existsSync(segment.filePath)) return { success: false, error: `Source file not found: ${path.basename(segment.filePath)}` }
    } catch (err) {
      return { success: false, error: String(err) }
    }
    const controller = new AbortController()
    activeExportController = controller
    try {
      return await runExportPipeline(input, controller.signal, { ...defaultDependencies, run: (_binary, args, options) => runFfmpeg(ffmpegPath, args, options), mix: (clips, duration, _binary, signal) => mixAudioToPcm(clips, duration, ffmpegPath, signal) })
    } finally {
      if (activeExportController === controller) activeExportController = null
    }
  })

  handle('exportCancel', () => {
    activeExportController?.abort()
    stopExportProcess()
    return { success: true }
  })
}
