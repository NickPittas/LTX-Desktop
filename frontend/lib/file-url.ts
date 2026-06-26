/**
 * Converts a filesystem path to a properly encoded file:// URL.
 *
 * Returns a native file:// URL for Electron/Chromium media playback.
 * Custom ltx-file:// demuxing was unreliable (MediaError code 4
 * PIPELINE_ERROR_READ for all codecs), so native file:// is used instead.
 *
 *   /Users/me/my file.mp4     → file:///Users/me/my%20file.mp4
 *   C:\Users\me\video#1.mp4   → file:///C:/Users/me/video%231.mp4
 */
export function pathToFileUrl(filePath: string): string {
  // Normalize Windows separators
  let normalized = filePath.replace(/\\/g, '/')

  // Ensure leading slash (Windows drive letters like C:/ need one prepended)
  if (!normalized.startsWith('/')) {
    normalized = '/' + normalized
  }

  // Encode each path segment, preserving Windows drive-letter colon
  const encoded = normalized
    .split('/')
    .map((segment) => {
      // Windows drive letter (e.g. C:) — keep the colon unencoded
      if (/^[A-Za-z]:$/.test(segment)) {
        return segment
      }
      return encodeURIComponent(segment)
    })
    .join('/')

  return 'file://' + encoded
}
