import type { components } from '../generated/backend-openapi'

export type OutputFormat = components['schemas']['OutputFormat']

export interface OutputFormatOption {
  value: OutputFormat
  label: string
  group: string
}

export const OUTPUT_FORMAT_OPTIONS: OutputFormatOption[] = [
  { value: 'mp4', label: 'MP4 (H.264)', group: 'Common' },
  { value: 'prores_proxy', label: 'ProRes Proxy', group: 'ProRes' },
  { value: 'prores_lt', label: 'ProRes LT', group: 'ProRes' },
  { value: 'prores_422', label: 'ProRes 422', group: 'ProRes' },
  { value: 'prores_422_hq', label: 'ProRes 422 HQ', group: 'ProRes' },
  { value: 'prores_4444', label: 'ProRes 4444', group: 'ProRes' },
  { value: 'prores_4444_xq', label: 'ProRes 4444 XQ', group: 'ProRes' },
  { value: 'exr_zip_half', label: 'EXR (half, ZIP)', group: 'EXR' },
  { value: 'exr_zip_float', label: 'EXR (float, ZIP)', group: 'EXR' },
]

export function isProResOrExrFormat(format: OutputFormat | undefined | null): boolean {
  if (!format) return false
  return format !== 'mp4'
}
