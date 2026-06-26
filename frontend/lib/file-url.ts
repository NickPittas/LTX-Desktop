/**
 * Converts a filesystem path to a properly encoded ltx-file:// URL.
 * Custom protocol bypasses Chromium file:// restrictions in production.
 *
 *   /Users/me/my file.mp4     → ltx-file:///Users/me/my%20file.mp4
 *   C:\Users\me\video#1.mp4   → ltx-file:///C:/Users/me/video%231.mp4
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

  return 'ltx-file://' + encoded
}
