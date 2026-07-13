export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context)
  } catch (error) {
    if (!specifier.startsWith('.') || /\.[^/]+$/.test(specifier)) throw error
    for (const extension of ['.ts', '.tsx']) {
      try {
        return await nextResolve(`${specifier}${extension}`, context)
      } catch {
        // Try the next TypeScript extension.
      }
    }
    throw error
  }
}
