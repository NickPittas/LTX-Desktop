import fs from 'fs'

type LogLevel = 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG'
type LogSource = 'Electron' | 'Renderer' | 'Backend'

let logFilePath: string | null = null

/** Called by initSessionLog() to tell the writer where to append. */
export function setLogFilePath(filePath: string): void {
  logFilePath = filePath
}

function formatTimestamp(): string {
  const now = new Date()
  const pad = (n: number, len = 2) => String(n).padStart(len, '0')
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())},${pad(now.getMilliseconds(), 3)}`
}

export function writeLog(level: LogLevel, source: LogSource, message: string): void {
  if (!logFilePath) {
    // Log file not yet initialized (early startup)
    return
  }

  const line = `${formatTimestamp()} - ${level} - [${source}] ${message}\n`
  try {
    fs.appendFileSync(logFilePath, line, 'utf-8')
  } catch {
    // Silently ignore write errors
  }
}

// Errno codes / message fragments produced when the stdout/stderr pipe is
// closed or gone (e.g. the launching terminal was quit, or the read end of the
// pipe was closed). A synchronous throw out of console.log / console.error in
// this case surfaces as an Electron "Uncaught Exception" crash dialog. The
// console mirror is best-effort; the session log file captures the message
// independently via writeLog (which writes to a file, not the pipe).
const STREAM_WRITE_ERROR_CODES = new Set(['EPIPE', 'EIO', 'ENOSPC', 'ENETRESET', 'ESHUTDOWN'])
const STREAM_WRITE_ERROR_TOKENS = [
  'epipe',
  'err_stream_destroyed',
  'err_stream_write_after_end',
]

function isStreamWriteError(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException | null | undefined)?.code
  if (typeof code === 'string' && STREAM_WRITE_ERROR_CODES.has(code)) {
    return true
  }
  const message = (err as { message?: unknown } | null | undefined)?.message
  if (typeof message === 'string') {
    const lower = message.toLowerCase()
    return STREAM_WRITE_ERROR_TOKENS.some((token) => lower.includes(token))
  }
  return false
}

/**
 * Best-effort console write that cannot crash the Electron main process.
 *
 * If stdout/stderr is a closed pipe (EPIPE / ERR_STREAM_DESTROYED / ...) the
 * underlying stream write throws synchronously. Stream-write failures are
 * swallowed silently (writeLog still captures the message). Any other failure
 * is recorded to the log file but never rethrown — logging is always non-fatal.
 * This only guards the console write itself; application exceptions are
 * untouched.
 */
export function safeConsole(consoleMethod: 'log' | 'warn' | 'error', ...args: unknown[]): void {
  try {
    console[consoleMethod](...args)
  } catch (err) {
    if (isStreamWriteError(err)) {
      return
    }
    try {
      writeLog('WARNING', 'Electron', `safeConsole: console.${consoleMethod} threw ${String(err)}`)
    } catch {
      // Even the log file is unavailable — there is nothing more to do. Never
      // throw out of a logging call path.
    }
  }
}

let streamGuardsInstalled = false

/**
 * Attaches idempotent 'error' listeners to process.stdout / process.stderr so
 * that an asynchronous EPIPE-style error emitted by either stream (after the
 * read end of the pipe is closed) cannot become an uncaught exception and
 * crash the main process. Stream errors are recorded to the log file but never
 * rethrown. This only handles stream errors; application exceptions are never
 * routed through these streams.
 */
export function guardProcessConsoleStreams(): void {
  if (streamGuardsInstalled) return
  streamGuardsInstalled = true

  const makeHandler = (label: string) => (err: unknown): void => {
    const detail = (err as NodeJS.ErrnoException | undefined)?.message ?? String(err)
    try {
      writeLog('WARNING', 'Electron', `${label} stream error suppressed: ${String(detail)}`)
    } catch {
      // Never throw from a stream error handler.
    }
  }

  if (process.stdout && typeof process.stdout.on === 'function') {
    process.stdout.on('error', makeHandler('stdout'))
  }
  if (process.stderr && typeof process.stderr.on === 'function') {
    process.stderr.on('error', makeHandler('stderr'))
  }
}

function log(level: LogLevel, consoleMethod: 'log' | 'warn' | 'error', message: string): void {
  safeConsole(consoleMethod, message)
  writeLog(level, 'Electron', message)
}

export const logger = {
  info: (message: string) => log('INFO', 'log', message),
  warn: (message: string) => log('WARNING', 'warn', message),
  error: (message: string) => log('ERROR', 'error', message),
  debug: (message: string) => log('DEBUG', 'log', message),
}

// Install stream-error guards as soon as the logger module is loaded by any
// main-process code. Idempotent; safe to also call via guardProcessConsoleStreams().
guardProcessConsoleStreams()
