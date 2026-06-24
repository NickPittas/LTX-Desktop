let cached: { url: string; token: string } | null = null

export async function getBackendCredentials(): Promise<{ url: string; token: string }> {
  if (!cached) cached = await window.electronAPI.getBackend()
  return cached
}

export function resetBackendCredentials(): void {
  cached = null
}

export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const { url, token } = await getBackendCredentials()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${url}${path}`, { ...init, headers })
}

export async function backendAdminFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {}
  new Headers(init?.headers).forEach((value, key) => { headers[key] = value })
  const result = await window.electronAPI.backendAdminRequest({
    path,
    method: (init?.method ?? 'GET').toUpperCase() as 'GET' | 'POST' | 'PATCH' | 'DELETE',
    headers,
    body: typeof init?.body === 'string' ? init.body : undefined,
  })
  return new Response(result.body, { status: result.status, statusText: result.statusText })
}

export async function backendWsUrl(path: string): Promise<string> {
  const { url, token } = await getBackendCredentials()
  const ws = url.replace('http://', 'ws://')
  const sep = path.includes('?') ? '&' : '?'
  return `${ws}${path}${sep}token=${token}`
}
