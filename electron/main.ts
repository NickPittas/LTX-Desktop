import './app-paths'
import { app, protocol } from 'electron'
import { createReadStream } from 'fs'
import { stat } from 'fs/promises'
import { extname } from 'path'
import { setupCSP } from './csp'
import { registerExportHandlers } from './export/export-handler'
import { stopExportProcess } from './export/ffmpeg-utils'
import { registerAppHandlers } from './ipc/app-handlers'
import { registerFileHandlers } from './ipc/file-handlers'
import { registerLogHandlers } from './ipc/log-handlers'
import { registerVideoProcessingHandlers } from './ipc/video-processing-handlers'
import { logger } from './logger'
import { initSessionLog } from './logging-management'
import { stopPythonBackend } from './python-backend'
import { initAutoUpdater } from './updater'
import { createWindow, getMainWindow } from './window'
import { sendAnalyticsEvent } from './analytics'

function logAppVersion(): void {
  if (!app.isPackaged) {
    logger.info('[LTX Desktop] Running in development mode')
  } else {
    logger.info(`[LTX Desktop] Version ${app.getVersion()}`)
  }
}

const MIME_TYPES: Record<string, string> = {
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
}

// ponytail: custom ltx-file:// protocol to serve local media files, bypassing Chromium file:// restrictions in production
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'ltx-file',
    privileges: { bypassCSP: false, stream: true, supportFetchAPI: true },
  },
])

const gotLock = app.requestSingleInstanceLock()

if (!gotLock) {
  app.quit()
} else {
  initSessionLog()
  logAppVersion()

  registerAppHandlers()
  registerFileHandlers()
  registerLogHandlers()
  registerExportHandlers()
  registerVideoProcessingHandlers()

  app.on('second-instance', () => {
    const mainWindow = getMainWindow()
    if (mainWindow) {
      if (mainWindow.isMinimized()) {
        mainWindow.restore()
      }
      if (!mainWindow.isVisible()) {
        mainWindow.show()
      }
      mainWindow.focus()
      return
    }
    if (app.isReady()) {
      createWindow()
    }
  })

  app.whenReady().then(async () => {
    // ponytail: custom ltx-file:// protocol — direct file streaming with Range support for video seeking
    protocol.handle('ltx-file', async (request) => {
      const filePath = decodeURIComponent(request.url.slice('ltx-file://'.length))
      const contentType = MIME_TYPES[extname(filePath).toLowerCase()] || 'application/octet-stream'

      try {
        const stats = await stat(filePath)
        const rangeHeader = request.headers.get('range')

        if (rangeHeader) {
          const match = rangeHeader.match(/bytes=(\d+)-(\d*)/)
          if (match) {
            const start = parseInt(match[1], 10)
            const end = match[2] ? parseInt(match[2], 10) : stats.size - 1
            const chunkSize = end - start + 1
            const nodeStream = createReadStream(filePath, { start, end })
            const webStream = new ReadableStream({
              start(controller) {
                nodeStream.on('data', (chunk: Buffer) => controller.enqueue(chunk))
                nodeStream.on('end', () => controller.close())
                nodeStream.on('error', (err) => controller.error(err))
              },
            })
            return new Response(webStream, {
              status: 206,
              headers: {
                'Content-Type': contentType,
                'Content-Length': String(chunkSize),
                'Content-Range': `bytes ${start}-${end}/${stats.size}`,
                'Accept-Ranges': 'bytes',
              },
            })
          }
        }

        // ponytail: full file stream (no Range header)
        const nodeStream = createReadStream(filePath)
        const webStream = new ReadableStream({
          start(controller) {
            nodeStream.on('data', (chunk: Buffer) => controller.enqueue(chunk))
            nodeStream.on('end', () => controller.close())
            nodeStream.on('error', (err) => controller.error(err))
          },
        })
        return new Response(webStream, {
          status: 200,
          headers: {
            'Content-Type': contentType,
            'Content-Length': String(stats.size),
            'Accept-Ranges': 'bytes',
          },
        })
      } catch {
        return new Response('File not found', { status: 404 })
      }
    })
    setupCSP()
    createWindow()
    initAutoUpdater()
    // Python setup + backend start are now driven by the renderer via IPC

    // Fire analytics event (no-op if user hasn't opted in)
    void sendAnalyticsEvent('ltxdesktop_app_launched')
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      stopPythonBackend()
      app.quit()
    }
  })

  app.on('activate', () => {
    if (getMainWindow() === null) {
      createWindow()
    }
  })

  app.on('before-quit', () => {
    stopExportProcess()
    stopPythonBackend()
  })
}
