import { ChildProcess, spawn } from 'child_process'
import crypto from 'crypto'
import fs from 'fs'
import path from 'path'
import { getAppDataDir } from './app-paths'
import { getCurrentDir, isDev } from './config'
import { HF_GATING_ENABLED } from '../shared/feature-flags'
import { logger, safeConsole, writeLog } from './logger'
import { getCurrentLogFilename } from './logging-management'
import { getPythonDir } from './python-setup'
import { getMainWindow } from './window'

let pythonProcess: ChildProcess | null = null
let isIntentionalShutdown = false
let lastCrashTime = 0
const CRASH_DEBOUNCE_MS = 10_000
let startPromise: Promise<void> | null = null
let takeoverInFlight: Promise<void> | null = null

// HTTP liveness monitoring: once the backend has answered /health after
// startup, poll it periodically. On sustained failure, SIGTERM the process so
// the exit handler runs the normal restart/dead flow.
const STARTUP_PROBE_TIMEOUT_MS = 30_000
const STARTUP_PROBE_INTERVAL_MS = 500
const LIVENESS_POLL_INTERVAL_MS = 10_000
const LIVENESS_FAILURE_THRESHOLD = 3
let livenessMonitorTimer: NodeJS.Timeout | null = null
let livenessFailureCount = 0

let backendUrl: string | null = null
let authToken: string | null = null
let adminToken: string | null = null

export function getBackendUrl(): string | null { return backendUrl }
export function getAuthToken(): string | null { return authToken }
export function getAdminToken(): string | null { return adminToken }

type BackendOwnership = 'managed' | 'adopted' | null

let backendOwnership: BackendOwnership = null

export interface BackendHealthStatus {
  status: 'alive' | 'restarting' | 'dead'
  exitCode?: number | null
}

let latestBackendHealthStatus: BackendHealthStatus | null = null

function publishBackendHealthStatus(status: BackendHealthStatus): void {
  latestBackendHealthStatus = status
  getMainWindow()?.webContents.send('backend-health-status', status)
}

export function getBackendHealthStatus(): BackendHealthStatus | null {
  return latestBackendHealthStatus
}

function getBackendPath(): string {
  if (isDev) {
    return path.join(getCurrentDir(), 'backend')
  }
  return path.join(process.resourcesPath, 'backend')
}

function isPortConflictOutput(output: string): boolean {
  const normalizedOutput = output.toLowerCase()
  return (
    normalizedOutput.includes('address already in use') ||
    normalizedOutput.includes('eaddrinuse') ||
    normalizedOutput.includes('errno 48')
  )
}

async function probeBackendHealth(timeoutMs = 1500, probeUrl?: string): Promise<boolean> {
  const url = probeUrl || backendUrl
  if (!url) return false
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const headers: Record<string, string> = {}
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`
    const response = await fetch(`${url}/health`, {
      signal: controller.signal,
      headers,
    })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

async function requestAdoptedBackendShutdown(timeoutMs = 2000): Promise<boolean> {
  if (!backendUrl) return false
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const headers: Record<string, string> = {}
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`
    const response = await fetch(`${backendUrl}/api/system/shutdown`, {
      method: 'POST',
      signal: controller.signal,
      headers,
    })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

async function waitUntilBackendDown(timeoutMs = 8000): Promise<boolean> {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    const healthy = await probeBackendHealth(800)
    if (!healthy) {
      return true
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  return false
}

function stopLivenessMonitor(): void {
  if (livenessMonitorTimer) {
    clearInterval(livenessMonitorTimer)
    livenessMonitorTimer = null
  }
  livenessFailureCount = 0
}

function startLivenessMonitor(): void {
  stopLivenessMonitor()
  livenessMonitorTimer = setInterval(() => {
    void (async () => {
      if (!pythonProcess || backendOwnership !== 'managed' || isIntentionalShutdown) {
        return
      }
      const healthy = await probeBackendHealth(2000)
      if (healthy) {
        livenessFailureCount = 0
        return
      }
      livenessFailureCount += 1
      logger.warn(`Backend liveness probe failed (${livenessFailureCount}/${LIVENESS_FAILURE_THRESHOLD})`)
      if (livenessFailureCount >= LIVENESS_FAILURE_THRESHOLD) {
        logger.error('Backend liveness probe failed repeatedly — killing process to trigger restart')
        stopLivenessMonitor()
        try {
          pythonProcess?.kill('SIGTERM')
        } catch {
          // Process may already be dead; exit handler will run.
        }
      }
    })()
  }, LIVENESS_POLL_INTERVAL_MS)
}

function startOwnershipTakeover(): void {
  if (takeoverInFlight || backendOwnership !== 'adopted') {
    return
  }

  takeoverInFlight = (async () => {
    try {
      const shutdownRequested = await requestAdoptedBackendShutdown()
      if (!shutdownRequested) {
        throw new Error('Failed to request shutdown for adopted backend')
      }

      const backendStopped = await waitUntilBackendDown()
      if (!backendStopped) {
        throw new Error('Timed out waiting for adopted backend shutdown')
      }

      backendOwnership = null
      await startPythonBackend()
    } catch (error) {
      logger.error(`Failed to reclaim backend process ownership: ${error}`)
      backendOwnership = null
      publishBackendHealthStatus({ status: 'dead' })
    } finally {
      takeoverInFlight = null
    }
  })()
}

export function getPythonPath(): string {
  // In production, use bundled/downloaded Python first
  if (!isDev) {
    const pythonDir = getPythonDir()
    const bundledPython = process.platform === 'win32'
      ? path.join(pythonDir, 'python.exe')
      : path.join(pythonDir, 'bin', 'python3')
    if (fs.existsSync(bundledPython)) {
      logger.info(`Using bundled Python: ${bundledPython}`)
      return bundledPython
    }
  }

  // Check for venv in backend directory
  const backendPath = getBackendPath()
  const isWindows = process.platform === 'win32'
  const venvPython = isWindows
    ? path.join(backendPath, '.venv', 'Scripts', 'python.exe')
    : path.join(backendPath, '.venv', 'bin', 'python')

  if (fs.existsSync(venvPython)) {
    logger.info(`Using venv Python: ${venvPython}`)
    return venvPython
  }

  if (isDev) {
    // In development, try common Python paths
    const pythonPaths = isWindows
      ? [
          'python',
          'python3',
          path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python311', 'python.exe'),
          path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python312', 'python.exe'),
        ]
      : [
          'python3',
          'python',
        ]

    for (const p of pythonPaths) {
      try {
        if (fs.existsSync(p)) {
          return p
        }
      } catch {
        continue
      }
    }
    return isWindows ? 'python' : 'python3'
  }

  // Fallback
  return 'python'
}

const BACKEND_PID_FILENAME = 'backend.pid'

function backendPidFilePath(): string {
  return path.join(getAppDataDir(), BACKEND_PID_FILENAME)
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (err) {
    // EPERM = the process exists but we may not signal it → still "alive".
    return (err as NodeJS.ErrnoException).code === 'EPERM'
  }
}

// Best-effort guard so we never kill an unrelated process that happens to have
// reused a stale PID. Linux: verify /proc/<pid>/cmdline references our server.
// Other platforms: we only ever record our own backend PID, so trust the file.
function isOurBackendProcess(pid: number): boolean {
  if (process.platform !== 'linux') return true
  try {
    const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8')
    return cmdline.includes('ltx2_server.py')
  } catch {
    return false
  }
}

function writeBackendPidFile(pid: number): void {
  try {
    fs.writeFileSync(backendPidFilePath(), String(pid), 'utf8')
  } catch {
    // Non-fatal: we just lose auto-reap for orphans left by this run.
  }
}

function clearBackendPidFile(): void {
  try {
    fs.rmSync(backendPidFilePath(), { force: true })
  } catch {
    // Non-fatal.
  }
}

// Terminate a backend orphaned by a previous session (unclean shutdown, crash,
// or a session that hung before it could clean up) so it can't keep holding the
// fixed backend port and block the new backend from binding. Only ever targets
// a PID we recorded ourselves and (on Linux) verified is our own server.
async function reapStaleBackend(): Promise<void> {
  let raw: string
  try {
    raw = fs.readFileSync(backendPidFilePath(), 'utf8').trim()
  } catch {
    return // No PID file → nothing to reap.
  }
  const pid = Number.parseInt(raw, 10)
  if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) {
    clearBackendPidFile()
    return
  }
  if (!isProcessAlive(pid) || !isOurBackendProcess(pid)) {
    clearBackendPidFile()
    return
  }

  logger.warn(`Reaping orphaned backend (pid ${pid}) left by a previous session to free the port`)
  try { process.kill(pid, 'SIGTERM') } catch { /* already gone */ }
  const graceDeadline = Date.now() + 3000
  while (isProcessAlive(pid) && Date.now() < graceDeadline) {
    await new Promise((resolveSleep) => setTimeout(resolveSleep, 150))
  }
  if (isProcessAlive(pid)) {
    try { process.kill(pid, 'SIGKILL') } catch { /* gone */ }
    const killDeadline = Date.now() + 2000
    while (isProcessAlive(pid) && Date.now() < killDeadline) {
      await new Promise((resolveSleep) => setTimeout(resolveSleep, 100))
    }
  }
  clearBackendPidFile()
}

export async function startPythonBackend(): Promise<void> {
  if (startPromise) {
    return startPromise
  }

  if (pythonProcess && backendOwnership === 'managed') {
    publishBackendHealthStatus({ status: 'alive' })
    return
  }

  if (backendOwnership === 'adopted') {
    const adoptedHealthy = await probeBackendHealth()
    if (adoptedHealthy) {
      publishBackendHealthStatus({ status: 'alive' })
      return
    }
    backendOwnership = null
  }

  isIntentionalShutdown = false

  // Clear any backend orphaned by a previous session before spawning, so it
  // can't hold the fixed port and make the new backend fail to bind.
  await reapStaleBackend()

  startPromise = new Promise((resolve, reject) => {
    const pythonPath = getPythonPath()
    const backendPath = getBackendPath()
    const mainPy = path.join(backendPath, 'ltx2_server.py')

    logger.info(`Starting Python backend: ${pythonPath} ${mainPy}`)

    // Windows embedded Python's ._pth file suppresses normal sys.path setup —
    // the script's directory isn't added, so sibling packages (e.g. state/)
    // can't be found. Use a -c wrapper to fix sys.path before running the server.
    let pythonArgs: string[]
    if (!isDev && process.platform === 'win32') {
      const preamble = `import sys; sys.path.insert(0, r"${backendPath}"); import runpy; runpy.run_path(r"${mainPy}", run_name="__main__")`
      pythonArgs = ['-u', '-c', preamble]
    } else {
      pythonArgs = isDev ? ['-Xfrozen_modules=off', '-u', mainPy] : ['-u', mainPy]
    }

    // Generate auth token and admin token for this backend session
    authToken = crypto.randomBytes(32).toString('base64url')
    adminToken = crypto.randomBytes(32).toString('base64url')

    pythonProcess = spawn(pythonPath, pythonArgs, {
      cwd: backendPath,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONNOUSERSITE: '1',
        PYDEVD_DISABLE_FILE_VALIDATION: '1',
        // Only pass LTX_PORT when the developer explicitly set it
        ...(process.env.LTX_PORT ? { LTX_PORT: process.env.LTX_PORT } : {}),
        LTX_AUTH_TOKEN: authToken,
        LTX_ADMIN_TOKEN: adminToken,
        LTX_LOG_FILE: getCurrentLogFilename(),
        LTX_APP_DATA_DIR: getAppDataDir(),
        LTX_DEV_MODE: isDev ? '1' : '0',
        LTX_HF_GATING_ENABLED: HF_GATING_ENABLED ? '1' : '0',
        PYTORCH_ENABLE_MPS_FALLBACK: '1',
        // Set PYTHONHOME for bundled Python on macOS so it finds its stdlib
        ...(!isDev && process.platform !== 'win32' ? {
          PYTHONHOME: getPythonDir(),
        } : {}),
      },
      stdio: ['ignore', 'pipe', 'pipe']
    })

    // Record the PID so a future session can reap this process if we exit
    // uncleanly (crash / kill / hung session) without freeing the port.
    if (pythonProcess.pid !== undefined) {
      writeBackendPidFile(pythonProcess.pid)
    }

    let started = false
    let startupSettled = false
    let sawPortConflict = false
    let probeGateStarted = false

    const settleResolve = () => {
      if (startupSettled) return
      startupSettled = true
      resolve()
    }

    const settleReject = (error: Error) => {
      if (startupSettled) return
      startupSettled = true
      reject(error)
    }

    const gateAliveOnProbe = async () => {
      const deadline = Date.now() + STARTUP_PROBE_TIMEOUT_MS
      while (Date.now() < deadline) {
        if (!pythonProcess || isIntentionalShutdown) {
          return
        }
        if (await probeBackendHealth(1500)) {
          started = true
          backendOwnership = 'managed'
          publishBackendHealthStatus({ status: 'alive' })
          settleResolve()
          startLivenessMonitor()
          return
        }
        await new Promise((resolveSleep) => setTimeout(resolveSleep, STARTUP_PROBE_INTERVAL_MS))
      }
      logger.error('Backend HTTP probe never succeeded after ready signal — killing process')
      try {
        pythonProcess?.kill('SIGTERM')
      } catch {
        // Exit handler will run and fail startup with dead.
      }
    }

    const checkStarted = (output: string) => {
      if (isPortConflictOutput(output)) {
        sawPortConflict = true
      }

      if (started || probeGateStarted) return

      // Capture the backend URL from EITHER uvicorn's "Uvicorn running on <url>"
      // or our "Server running on <url>" line — whichever arrives first in the
      // stdout stream (they can land in separate chunks). Both carry the URL.
      // Previously only "Server running on" set backendUrl, and a separate-chunk
      // "Uvicorn running" line flipped started=true via a URL-less fallback,
      // leaving the renderer with no backend URL (stuck on "Loading settings").
      // gateAliveOnProbe HTTP-probes /health, so 'alive' is still only published
      // once the server actually answers.
      const readyMatch = output.match(/(?:Server|Uvicorn) running on (http:\/\/\S+)/)
      if (readyMatch) {
        backendUrl = readyMatch[1]
        probeGateStarted = true
        void gateAliveOnProbe()
      }
    }

    pythonProcess.stdout?.on('data', (data: Buffer) => {
      const output = data.toString()
      safeConsole('log', `[Python] ${output}`)
      for (const line of output.split('\n')) {
        const trimmed = line.trimEnd()
        if (trimmed) writeLog('INFO', 'Backend', trimmed)
      }
      checkStarted(output)
    })

    pythonProcess.stderr?.on('data', (data: Buffer) => {
      const output = data.toString()
      safeConsole('error', `[Python Error] ${output}`)
      for (const line of output.split('\n')) {
        const trimmed = line.trimEnd()
        if (trimmed) writeLog('ERROR', 'Backend', trimmed)
      }
      checkStarted(output)
    })

    pythonProcess.on('error', (error) => {
      logger.error(`Failed to start Python backend: ${error}`)
      if (!started) {
        backendOwnership = null
        publishBackendHealthStatus({ status: 'dead' })
        settleReject(error)
      }
    })

    pythonProcess.on('exit', async (code) => {
      logger.info(`Python backend exited with code ${code}`)
      stopLivenessMonitor()
      pythonProcess = null
      backendUrl = null
      authToken = null
      adminToken = null

      if (!started) {
        if (isIntentionalShutdown) {
          isIntentionalShutdown = false
          backendOwnership = null
          settleReject(new Error('Python backend stopped during startup'))
          return
        }

        if (sawPortConflict && process.env.LTX_PORT) {
          const explicitUrl = `http://127.0.0.1:${process.env.LTX_PORT}`
          const healthyExistingBackend = await probeBackendHealth(1500, explicitUrl)
          if (healthyExistingBackend) {
            backendUrl = explicitUrl
            backendOwnership = 'adopted'
            publishBackendHealthStatus({ status: 'alive' })
            settleResolve()
            startOwnershipTakeover()
            return
          }
        }

        backendOwnership = null
        publishBackendHealthStatus({ status: 'dead', exitCode: code })
        settleReject(new Error(`Python backend exited during startup with code ${code}`))
        return
      }

      if (isIntentionalShutdown) {
        isIntentionalShutdown = false
        backendOwnership = null
        return
      }

      backendOwnership = 'managed'
      const now = Date.now()
      if (now - lastCrashTime < CRASH_DEBOUNCE_MS) {
        publishBackendHealthStatus({ status: 'dead', exitCode: code })
        return
      }

      lastCrashTime = now
      publishBackendHealthStatus({ status: 'restarting', exitCode: code })
      try {
        await startPythonBackend()
      } catch {
        publishBackendHealthStatus({ status: 'dead', exitCode: code })
      }
    })

    // Timeout after 5 minutes (model loading can take a while on first run)
    setTimeout(() => {
      if (startupSettled || started) {
        return
      }

      try {
        pythonProcess?.kill('SIGTERM')
      } catch {
        // Process may already be dead.
      }
      backendOwnership = null
      publishBackendHealthStatus({ status: 'dead' })
      settleReject(new Error('Python backend failed to start within 5 minutes'))
    }, 300000)
  })

  try {
    await startPromise
  } finally {
    startPromise = null
  }
}

export function stopPythonBackend(): void {
  if (pythonProcess) {
    isIntentionalShutdown = true
    stopLivenessMonitor()
    clearBackendPidFile()
    logger.info('Stopping Python backend...')
    const pid = pythonProcess.pid
    pythonProcess.kill('SIGTERM')
    pythonProcess = null
    // Force kill after 5 seconds if SIGTERM didn't work (PyTorch/uvicorn threads)
    if (pid) {
      setTimeout(() => {
        try {
          process.kill(pid, 0) // Check if still alive (throws if dead)
          process.kill(pid, 'SIGKILL')
        } catch {
          // Already dead
        }
      }, 5000)
    }
    return
  }

  if (backendOwnership === 'adopted') {
    backendOwnership = null
    latestBackendHealthStatus = null
  }
}
