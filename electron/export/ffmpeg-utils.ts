import { spawn, spawnSync, ChildProcess, execSync } from 'child_process'
import os from 'os'
import path from 'path'
import fs from 'fs'
import { isDev, getCurrentDir } from '../config'
import { logger } from '../logger'
import { getPythonDir } from '../python-setup'

let activeExportProcess: ChildProcess | null = null

export function findFfmpegPath(): string | null {
  let binDir: string | null = null

  if (process.platform === 'win32') {
    const imageioRelPath = path.join('Lib', 'site-packages', 'imageio_ffmpeg', 'binaries')
    binDir = isDev
      ? path.join(getCurrentDir(), 'backend', '.venv', imageioRelPath)
      : path.join(getPythonDir(), imageioRelPath)
  } else {
    // macOS/Linux: find lib/python3.X/site-packages dynamically
    const venvBase = isDev
      ? path.join(getCurrentDir(), 'backend', '.venv')
      : getPythonDir()
    const libDir = path.join(venvBase, 'lib')
    if (fs.existsSync(libDir)) {
      const pythonDir = fs.readdirSync(libDir).find(e => e.startsWith('python3'))
      if (pythonDir) {
        binDir = path.join(libDir, pythonDir, 'site-packages', 'imageio_ffmpeg', 'binaries')
      }
    }
  }

  if (binDir && fs.existsSync(binDir)) {
    const bin = fs.readdirSync(binDir).find(f => f.startsWith('ffmpeg'))
    if (bin) return path.join(binDir, bin)
  }

  try { execSync('ffmpeg -version', { stdio: 'ignore' }); return 'ffmpeg' } catch { return null }
}

/** Check if a video file contains an audio stream using ffprobe/ffmpeg */
export function fileHasAudio(ffmpegPath: string, filePath: string): boolean {
  try {
    const result = spawnSync(ffmpegPath, ['-i', filePath, '-hide_banner'], {
      encoding: 'utf8',
      timeout: 5000,
    })
    const output = (result.stdout || '') + (result.stderr || '')
    return output.includes('Audio:')
  } catch {
    return false
  }
}

/**
 * Probe the duration (seconds) of a media file using only the bundled ffmpeg
 * binary (imageio ships ffmpeg, not ffprobe). Returns null if it can't be
 * determined. Used to convert ffmpeg `-progress` `out_time_us` into a 0..1
 * fraction for import-transcode progress reporting.
 */
export function getMediaDurationSeconds(mediaPath: string): number | null {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath || !fs.existsSync(mediaPath)) return null
  try {
    const result = spawnSync(ffmpegPath, ['-hide_banner', '-i', mediaPath], {
      encoding: 'utf8',
      timeout: 10000,
    })
    const output = `${result.stdout || ''}\n${result.stderr || ''}`
    const m = output.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/)
    if (!m) return null
    const h = Number(m[1])
    const min = Number(m[2])
    const sec = Number(m[3])
    if (![h, min, sec].every(Number.isFinite)) return null
    return h * 3600 + min * 60 + sec
  } catch {
    return null
  }
}

export interface RunFfmpegOptions {
  /**
   * Progress callback receiving a fraction in [0, 1]. When provided, ffmpeg is
   * launched with `-progress pipe:1` and its key=value stdout stream is parsed.
   */
  onProgress?: (pct: number) => void
  /**
   * When true, the child process is NOT registered in the global
   * `activeExportProcess`, so export-cancel cannot kill it. Use for import /
   * background transcodes. Export calls omit this (default false) and remain
   * cancellable — byte-for-byte current behavior.
   */
  isolated?: boolean
  /** Input duration in microseconds; used to compute pct from `out_time_us`. */
  durationUs?: number
  /** Total-frames hint; used to compute pct from the `frame` count. */
  totalFrames?: number
}

/**
 * Run an ffmpeg command and return a promise. Logs stderr and (for non-isolated
 * runs) sets `activeExportProcess` so export-cancel can kill it.
 *
 * The 2-arg form (no options) is the export path and is unchanged: the process
 * is registered in `activeExportProcess`, stderr is logged, and stdout is left
 * unread. Passing `options` only affects import/background usage.
 */
export function runFfmpeg(
  ffmpegPath: string,
  args: string[],
  options?: RunFfmpegOptions,
): Promise<{ success: boolean; error?: string }> {
  const onProgress = options?.onProgress
  const isolated = options?.isolated === true
  const durationUs = options?.durationUs
  const totalFrames = options?.totalFrames
  const trackProgress = typeof onProgress === 'function'

  return new Promise((resolve) => {
    // `-progress pipe:1` emits key=value update blocks on stdout. Prepend it so
    // the rest of the arg vector is untouched.
    const finalArgs = trackProgress ? ['-progress', 'pipe:1', ...args] : args
    logger.info(`[ffmpeg] spawn: ${finalArgs.join(' ').slice(0, 400)}`)
    const proc = spawn(ffmpegPath, finalArgs, { stdio: ['pipe', 'pipe', 'pipe'] })

    // Only the non-isolated (export) path registers for cancel-kill.
    if (!isolated) {
      activeExportProcess = proc
    }

    let stderrLog = ''
    proc.stderr?.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      stderrLog += text
      const lines = text.trim().split('\n')
      for (const line of lines) {
        if (line.includes('frame=') || line.includes('Error') || line.includes('error')) {
          logger.info(`[ffmpeg] ${line.trim().slice(0, 200)}`)
        }
      }
    })

    if (trackProgress && proc.stdout) {
      // Parse ffmpeg `-progress` stdout: key=value lines, one update block
      // terminated by `progress=continue` (or `progress=end` at the very end).
      let stdoutBuf = ''
      const fields = new Map<string, string>()
      let lastEmitTs = 0
      const MIN_EMIT_INTERVAL_MS = 100 // ≤ 10 updates/sec

      const computeFraction = (): number => {
        const outTimeUs = Number(fields.get('out_time_us'))
        if (durationUs && durationUs > 0 && Number.isFinite(outTimeUs) && outTimeUs >= 0) {
          return Math.min(1, outTimeUs / durationUs)
        }
        const frame = Number(fields.get('frame'))
        if (totalFrames && totalFrames > 0 && Number.isFinite(frame) && frame >= 0) {
          return Math.min(1, frame / totalFrames)
        }
        return -1
      }

      const flush = (force: boolean): void => {
        const frac = computeFraction()
        if (frac < 0) return
        const now = Date.now()
        if (!force && now - lastEmitTs < MIN_EMIT_INTERVAL_MS) return
        lastEmitTs = now
        try {
          onProgress(Math.max(0, Math.min(1, frac)))
        } catch {
          /* ignore renderer-side callback errors */
        }
      }

      proc.stdout.on('data', (chunk: Buffer) => {
        stdoutBuf += chunk.toString()
        let nl = stdoutBuf.indexOf('\n')
        while (nl >= 0) {
          const line = stdoutBuf.slice(0, nl).trim()
          stdoutBuf = stdoutBuf.slice(nl + 1)
          nl = stdoutBuf.indexOf('\n')
          if (!line) continue
          const eq = line.indexOf('=')
          if (eq <= 0) continue
          const key = line.slice(0, eq)
          const value = line.slice(eq + 1)
          fields.set(key, value)
          // `progress=` terminates an update block — flush a throttled update.
          if (key === 'progress') {
            flush(false)
          }
        }
      })
    }

    proc.on('close', (code) => {
      // Only the non-isolated path owns the global slot; never clear it from an
      // isolated run (it may be holding a different, in-flight export process).
      if (!isolated) {
        activeExportProcess = null
      }
      if (code === 0) {
        if (trackProgress) {
          try {
            onProgress(1)
          } catch {
            /* ignore */
          }
        }
        resolve({ success: true })
      } else {
        const errLines = stderrLog.split('\n').filter(l => l.trim()).slice(-5).join('\n')
        logger.error(`[ffmpeg] exited ${code}:\n${errLines}`)
        resolve({ success: false, error: `FFmpeg failed (code ${code}): ${errLines.slice(0, 300)}` })
      }
    })
    proc.on('error', (err) => {
      if (!isolated) {
        activeExportProcess = null
      }
      resolve({ success: false, error: `Failed to start ffmpeg: ${err.message}` })
    })
  })
}

function runFfmpegSyncOrThrow(ffmpegPath: string, args: string[], timeoutMs = 30000): void {
  logger.info(`[ffmpeg-sync] spawn: ${args.join(' ').slice(0, 400)}`)
  const result = spawnSync(ffmpegPath, args, { timeout: timeoutMs })
  if (result.status === 0) return
  const stderr = (result.stderr?.toString() || '').split('\n').filter(Boolean).slice(-5).join('\n')
  throw new Error(`FFmpeg failed (code ${result.status}): ${stderr.slice(0, 300)}`)
}

export function extractVideoFrameToFile({
  videoPath,
  seekTime,
  width,
  quality,
  outputPath,
  timeoutMs = 10000,
}: {
  videoPath: string
  seekTime: number
  width?: number
  quality?: number
  outputPath?: string
  timeoutMs?: number
}): string {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) {
    throw new Error('ffmpeg not found')
  }
  if (!fs.existsSync(videoPath)) {
    throw new Error(`Video file not found: ${videoPath}`)
  }

  const resolvedOutputPath = outputPath
    ?? path.join(
      os.tmpdir(),
      `ltx_frame_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.jpg`,
    )

  const args: string[] = [
    '-ss', String(Math.max(0, seekTime)),
    '-i', videoPath,
    ...(width ? ['-vf', `scale=${width}:-2`] : []),
    '-frames:v', '1',
    ...(quality !== undefined ? ['-q:v', String(quality)] : []),
    '-y',
    resolvedOutputPath,
  ]

  logger.info(`[extract-frame] ${args.join(' ').slice(0, 300)}`)
  runFfmpegSyncOrThrow(ffmpegPath, args, timeoutMs)

  if (!fs.existsSync(resolvedOutputPath)) {
    throw new Error('ffmpeg produced no output file')
  }

  return resolvedOutputPath
}

export function getVideoDimensions(videoPath: string): { width: number; height: number } {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) {
    throw new Error('ffmpeg not found')
  }
  if (!fs.existsSync(videoPath)) {
    throw new Error(`Video file not found: ${videoPath}`)
  }

  const result = spawnSync(ffmpegPath, ['-hide_banner', '-i', videoPath], {
    encoding: 'utf8',
    timeout: 10000,
  })
  const output = `${result.stdout || ''}\n${result.stderr || ''}`
  const videoStreamLine = output.split('\n').find(line => line.includes('Video:'))
  const match = videoStreamLine?.match(/(\d{2,5})x(\d{2,5})(?:[,\s\[]|$)/)

  if (!match) {
    throw new Error(`Could not determine video dimensions for ${videoPath}`)
  }

  const width = Number(match[1])
  const height = Number(match[2])
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error(`Invalid video dimensions for ${videoPath}: ${match[1]}x${match[2]}`)
  }

  return { width, height }
}

export function stopExportProcess(): void {
  if (activeExportProcess) {
    logger.info( 'Stopping active export process...')
    activeExportProcess.kill()
    activeExportProcess = null
  }
}
