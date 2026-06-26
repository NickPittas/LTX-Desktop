import { app, dialog } from 'electron'
import path from 'path'
import fs from 'fs'
import { checkGPU } from '../gpu'
import { isPythonReady, downloadPythonEmbed } from '../python-setup'
import { getBackendHealthStatus, getBackendUrl, getAuthToken, getAdminToken, startPythonBackend } from '../python-backend'
import { getMainWindow } from '../window'
import { getAnalyticsState, setAnalyticsEnabled, sendAnalyticsEvent } from '../analytics'
import { handle } from './typed-handle'

function writeSettingsFile(settingsPath: string, settings: Record<string, unknown>): void {
  // ponytail: ensure parent dir exists before write
  // upgrade path: shared setup-state service if this grows
  const dir = path.dirname(settingsPath)
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }
  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2))
}

function getModelsPath(): string {
  const modelsPath = path.join(app.getPath('userData'), 'models')
  if (!fs.existsSync(modelsPath)) {
    fs.mkdirSync(modelsPath, { recursive: true })
  }
  return modelsPath
}

function getSetupStatus(settingsPath: string): { needsSetup: boolean; needsLicense: boolean } {
  if (!fs.existsSync(settingsPath)) {
    // ponytail: if user has active local model profile but no app_state,
    // consider setup done. upgrade path: shared setup-state service.
    const profilesPath = path.join(path.dirname(settingsPath), 'model_profiles.json')
    if (fs.existsSync(profilesPath)) {
      try {
        const raw = JSON.parse(fs.readFileSync(profilesPath, 'utf-8'))
        const activeId = raw.active_model_profile_id
        const hasActiveProfile = activeId && Array.isArray(raw.profiles) &&
          raw.profiles.some((p: Record<string, unknown>) => p.id === activeId && p.isActive)
        if (hasActiveProfile) {
          return { needsSetup: false, needsLicense: true }
        }
      } catch {
        // Corrupt profiles file — proceed with full setup
      }
    }
    return { needsSetup: true, needsLicense: true }
  }
  try {
    const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    return {
      needsSetup: !settings.setupComplete,
      needsLicense: !settings.licenseAccepted,
    }
  } catch {
    return { needsSetup: true, needsLicense: true }
  }
}

function markSetupComplete(settingsPath: string): void {
  let settings: Record<string, unknown> = {}

  try {
    if (fs.existsSync(settingsPath)) {
      settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    }
  } catch {
    settings = {}
  }

  settings.setupComplete = true
  settings.licenseAccepted = true
  settings.licenseAcceptedDate = new Date().toISOString()
  settings.setupDate = new Date().toISOString()

  writeSettingsFile(settingsPath, settings)
}

function markLicenseAccepted(settingsPath: string): void {
  let settings: Record<string, unknown> = {}

  try {
    if (fs.existsSync(settingsPath)) {
      settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    }
  } catch {
    settings = {}
  }

  settings.licenseAccepted = true
  settings.setupComplete = true
  settings.licenseAcceptedDate = new Date().toISOString()
  settings.setupDate = new Date().toISOString()

  writeSettingsFile(settingsPath, settings)
}

export function registerAppHandlers(): void {
  handle('getBackend', () => {
    return { url: getBackendUrl() ?? '', token: getAuthToken() ?? '' }
  })

  handle('backendAdminRequest', async ({ path, method, headers, body }) => {
    const url = getBackendUrl()
    const auth = getAuthToken()
    const admin = getAdminToken()
    if (!url || !auth || !admin) return { status: 503, statusText: 'Unavailable', ok: false, body: 'Backend not ready' }

    const baseUrl = new URL(url)
    const targetUrl = new URL(path, baseUrl)
    const allowed = targetUrl.origin === baseUrl.origin && (
      targetUrl.pathname === '/api/model-profiles'
      || targetUrl.pathname.startsWith('/api/model-profiles/')
      || targetUrl.pathname === '/api/models/adapters/status'
      || targetUrl.pathname === '/api/models/adapters/recommendation'
    )
    if (!allowed) return { status: 403, statusText: 'Forbidden', ok: false, body: 'Admin path not allowed' }

    const requestHeaders = new Headers(headers)
    requestHeaders.set('Authorization', `Bearer ${auth}`)
    requestHeaders.set('X-Admin-Token', admin)
    const resp = await fetch(targetUrl, { method, headers: requestHeaders, body })
    return { status: resp.status, statusText: resp.statusText, ok: resp.ok, body: await resp.text() }
  })

  handle('getModelsPath', () => {
    return getModelsPath()
  })

  handle('checkGpu', async () => {
    return await checkGPU()
  })

  handle('getAppInfo', () => {
    return {
      version: app.getVersion(),
      isPackaged: app.isPackaged,
      modelsPath: getModelsPath(),
      userDataPath: app.getPath('userData'),
    }
  })

  handle('getDownloadsPath', () => {
    return app.getPath('downloads')
  })

  handle('checkFirstRun', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    return getSetupStatus(settingsPath)
  })

  handle('acceptLicense', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markLicenseAccepted(settingsPath)
    return true
  })

  handle('completeSetup', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markSetupComplete(settingsPath)
    return true
  })

  handle('fetchLicenseText', async () => {
    const resp = await fetch('https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE')
    if (!resp.ok) {
      throw new Error(`Failed to fetch license (HTTP ${resp.status})`)
    }
    return await resp.text()
  })

  handle('getNoticesText', async () => {
    const noticesPath = path.join(app.getAppPath(), 'NOTICES.md')
    return fs.readFileSync(noticesPath, 'utf-8')
  })

  handle('getResourcePath', () => {
    if (!app.isPackaged) {
      return null
    }
    return process.resourcesPath
  })

  handle('checkPythonReady', () => {
    return isPythonReady()
  })

  handle('startPythonSetup', async () => {
    await downloadPythonEmbed((progress) => {
      getMainWindow()?.webContents.send('python-setup-progress', progress)
    })
  })

  handle('startPythonBackend', async () => {
    await startPythonBackend()
  })

  handle('getBackendHealthStatus', () => {
    return getBackendHealthStatus()
  })

  handle('getAnalyticsState', () => {
    return getAnalyticsState()
  })

  handle('setAnalyticsEnabled', ({ enabled }) => {
    setAnalyticsEnabled(enabled)
  })

  handle('sendAnalyticsEvent', async ({ eventName, extraDetails }) => {
    await sendAnalyticsEvent(eventName, extraDetails)
  })

  handle('openModelsDirChangeDialog', async () => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return { success: false, error: 'No window' }

    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Select Models Directory',
      properties: ['openDirectory', 'createDirectory'],
    })
    if (result.canceled || !result.filePaths.length) return { success: false, error: 'cancelled' }

    const newDir = result.filePaths[0]
    const url = getBackendUrl()
    const auth = getAuthToken()
    const admin = getAdminToken()
    if (!url || !auth || !admin) return { success: false, error: 'Backend not ready' }

    const resp = await fetch(`${url}/api/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth}`,
        'X-Admin-Token': admin,
      },
      body: JSON.stringify({ modelsDir: newDir }),
    })
    if (!resp.ok) return { success: false, error: await resp.text() }

    return { success: true, path: newDir }
  })

}
