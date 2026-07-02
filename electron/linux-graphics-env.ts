import { existsSync } from 'fs'

/**
 * ponytail: Apply VAAPI env defaults for Linux + NVIDIA systems before Electron imports.
 *
 * On Fedora KDE Wayland with NVIDIA, Chromium/Electron logs `vaInitialize failed`
 * unless launched with `LIBVA_DRIVER_NAME=nvidia NVD_BACKEND=direct`. We set those
 * defaults here, before `electron` is imported, so they propagate to the GPU process.
 *
 * Rules:
 * - Non-Linux: do nothing.
 * - No NVIDIA detected: do nothing.
 * - User-provided env wins: never override `LIBVA_DRIVER_NAME` or `NVD_BACKEND`.
 * - No Chromium flags are toggled; we don't disable GPU, VAAPI, or Wayland.
 *
 * This module must be imported before anything that imports `electron` (e.g. before
 * `./app-paths`) so the env vars are set prior to app/GPU process initialization.
 */

function isNvidiaPresent(): boolean {
  // 1. Explicit env hint from the user / display stack.
  if (process.env.LIBVA_DRIVER_NAME === 'nvidia') return true
  if (process.env.__GLX_VENDOR_LIBRARY_NAME === 'nvidia') return true
  if (process.env.GBM_BACKEND === 'nvidia-drm') return true

  // 2. /proc/driver/nvidia/version exists on systems with the NVIDIA driver loaded.
  try {
    return existsSync('/proc/driver/nvidia/version')
  } catch {
    return false
  }
}

function applyLinuxNvidiaVaapiDefaults(): void {
  if (process.platform !== 'linux') return
  if (!isNvidiaPresent()) return

  if (!process.env.LIBVA_DRIVER_NAME) {
    process.env.LIBVA_DRIVER_NAME = 'nvidia'
  }
  if (!process.env.NVD_BACKEND) {
    process.env.NVD_BACKEND = 'direct'
  }
}

applyLinuxNvidiaVaapiDefaults()
