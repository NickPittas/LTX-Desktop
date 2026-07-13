import path from 'path'
import type fs from 'fs'
import { flattenTimeline, type ExportClip } from './timeline'
import { buildVideoFilterGraph, type ExportSubtitle } from './video-filter'
import { buildExportVideoPlan } from './export-codec'

export interface ExportInput {
  clips: ExportClip[]; outputPath: string; codec: string; width: number; height: number; fps: number; quality: number
  letterbox?: { ratio: number; color: string; opacity: number }; subtitles?: ExportSubtitle[]
}

export type ExportResult = { success: true } | { success: false; error: string }

export interface ExportRunnerDependencies {
  run: (binary: string, args: string[], options?: { signal?: AbortSignal }) => Promise<{ success: boolean; error?: string }>
  mix: (clips: ExportClip[], totalDuration: number, ffmpegPath: string, signal?: AbortSignal) => Promise<{ pcmBuffer: Buffer; sampleRate: number; channels: number }>
  writeFile: typeof fs.writeFileSync
  unlink: (filePath: string) => void
  rename: typeof fs.renameSync
  tmpDir: string
  unique: () => string
}

function cancelled(signal: AbortSignal): ExportResult | null {
  return signal.aborted ? { success: false, error: 'Export cancelled' } : null
}

export async function runExportPipeline(input: ExportInput, signal: AbortSignal, deps: ExportRunnerDependencies): Promise<ExportResult> {
  const plan = buildExportVideoPlan(input.codec, input.quality)
  const segments = flattenTimeline(input.clips)
  if (segments.length === 0) return { success: false, error: 'No clips to export' }
  const id = deps.unique()
  const tmpVideo = path.join(deps.tmpDir, `ltx-export-video-${id}${plan.tempExtension}`)
  const tmpAudio = path.join(deps.tmpDir, `ltx-export-audio-${id}.wav`)
  const tmpRawPcm = path.join(deps.tmpDir, `ltx-pcm-${id}.raw`)
  const filterFile = path.join(deps.tmpDir, `ltx-filter-v-${id}.txt`)
  const parsedOutput = path.parse(input.outputPath)
  const pendingFinal = path.join(parsedOutput.dir, `${parsedOutput.name}.ltx-pending-${id}${plan.tempExtension}`)
  const ownedPaths = [tmpVideo, tmpAudio, tmpRawPcm, filterFile, pendingFinal]

  try {
    if (cancelled(signal)) return cancelled(signal)!
    const { inputs, filterScript } = buildVideoFilterGraph(segments, { width: input.width, height: input.height, fps: input.fps, targetPixFmt: plan.targetPixFmt, letterbox: input.letterbox, subtitles: input.subtitles })
    deps.writeFile(filterFile, filterScript, 'utf8')
    let result = await deps.run('ffmpeg', ['-y', ...inputs, '-filter_complex_script', filterFile, '-map', '[outv]', '-an', ...plan.pass1VideoArgs, tmpVideo], { signal })
    if (!result.success) return { success: false, error: result.error ?? 'FFmpeg failed' }
    if (cancelled(signal)) return cancelled(signal)!

    let totalDuration = segments.reduce((max, segment) => Math.max(max, segment.startTime + segment.duration), 0)
    for (const clip of input.clips) totalDuration = Math.max(totalDuration, clip.startTime + clip.duration)
    const { pcmBuffer, sampleRate, channels } = await deps.mix(input.clips, totalDuration, 'ffmpeg', signal)
    if (cancelled(signal)) return cancelled(signal)!
    deps.writeFile(tmpRawPcm, pcmBuffer)
    result = await deps.run('ffmpeg', ['-y', '-f', 's16le', '-ar', String(sampleRate), '-ac', String(channels), '-i', tmpRawPcm, '-c:a', 'pcm_s16le', tmpAudio], { signal })
    if (!result.success) return { success: false, error: result.error ?? 'FFmpeg failed' }
    if (cancelled(signal)) return cancelled(signal)!

    result = await deps.run('ffmpeg', ['-y', '-i', tmpVideo, '-i', tmpAudio, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', ...plan.finalAudioArgs, '-f', plan.muxer, ...plan.finalContainerArgs, '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-shortest', pendingFinal], { signal })
    if (!result.success) return { success: false, error: result.error ?? 'FFmpeg failed' }
    if (cancelled(signal)) return cancelled(signal)!
    deps.rename(pendingFinal, input.outputPath)
    return { success: true }
  } catch (err) {
    return { success: false, error: signal.aborted ? 'Export cancelled' : String(err) }
  } finally {
    for (const filePath of ownedPaths) deps.unlink(filePath)
  }
}
