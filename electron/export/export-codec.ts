export interface ExportVideoPlan {
  codec: 'h264' | 'prores' | 'vp9'
  targetPixFmt: 'yuv420p' | 'yuv422p10le'
  tempExtension: '.mkv' | '.mov' | '.webm'
  muxer: 'mp4' | 'mov' | 'webm'
  pass1VideoArgs: string[]
  finalAudioArgs: string[]
  finalContainerArgs: string[]
}

export function buildExportVideoPlan(codec: string, quality: number): ExportVideoPlan {
  if (codec === 'h264') {
    if (!Number.isInteger(quality) || quality < 0 || quality > 51) throw new Error('H.264 quality must be an integer CRF from 0 to 51')
    return { codec, targetPixFmt: 'yuv420p', tempExtension: '.mkv', muxer: 'mp4', pass1VideoArgs: ['-c:v', 'libx264', '-preset', 'fast', '-crf', String(quality), '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709'], finalAudioArgs: ['-c:a', 'aac', '-b:a', '192k'], finalContainerArgs: ['-movflags', '+faststart'] }
  }
  if (codec === 'prores') {
    if (!Number.isInteger(quality) || quality < 0 || quality > 3) throw new Error('ProRes quality must be an integer profile from 0 to 3')
    return { codec, targetPixFmt: 'yuv422p10le', tempExtension: '.mov', muxer: 'mov', pass1VideoArgs: ['-c:v', 'prores_ks', '-profile:v', String(quality), '-pix_fmt', 'yuv422p10le', '-vendor', 'apl0', '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709'], finalAudioArgs: ['-c:a', 'pcm_s16le'], finalContainerArgs: [] }
  }
  if (codec === 'vp9') {
    if (!Number.isFinite(quality) || quality <= 0) throw new Error('VP9 quality must be a positive Mbps value')
    return { codec, targetPixFmt: 'yuv420p', tempExtension: '.webm', muxer: 'webm', pass1VideoArgs: ['-c:v', 'libvpx-vp9', '-b:v', `${quality}M`, '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709'], finalAudioArgs: ['-c:a', 'libopus', '-b:a', '128k'], finalContainerArgs: [] }
  }
  throw new Error(`Unknown codec: ${codec}`)
}
