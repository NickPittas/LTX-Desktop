import assert from 'node:assert/strict'
import test from 'node:test'
import { runExportPipeline, type ExportRunnerDependencies } from './export-runner'
import type { ExportClip } from './timeline'

const clip: ExportClip = { path: '/primary.mov', audioPath: '/proxy.mp4', type: 'video', startTime: 0, duration: 1, trimStart: 0, speed: 1, reversed: false, flipH: false, flipV: false, opacity: 100, trackIndex: 0, muted: false, volume: 1 }

function dependencies(run: ExportRunnerDependencies['run'], mix: ExportRunnerDependencies['mix'], unlinked: string[]): ExportRunnerDependencies {
  return { run, mix, writeFile: () => {}, unlink: filePath => unlinked.push(filePath), rename: () => {}, tmpDir: '/tmp', unique: () => 'test' }
}

test('proxy_audio_is_used_without_replacing_primary_video', async () => {
  const calls: string[][] = []
  let mixed: ExportClip[] = []
  const result = await runExportPipeline({ clips: [clip], outputPath: '/output.mov', codec: 'prores', width: 16, height: 16, fps: 24, quality: 3 }, new AbortController().signal, dependencies(async (_bin, args) => { calls.push(args); return { success: true } }, async clips => { mixed = clips; return { pcmBuffer: Buffer.alloc(0), sampleRate: 48000, channels: 2 } }, []))
  assert.deepEqual(result, { success: true })
  assert.ok(calls[0].includes('/primary.mov'))
  assert.equal(mixed[0].audioPath, '/proxy.mp4')
  assert.deepEqual(calls[2].slice(9, 11), ['-c:v', 'copy'])
})

test('cancellation_stops_later_passes_and_cleans_all_owned_paths', async () => {
  const controller = new AbortController()
  const unlinked: string[] = []
  let calls = 0
  const result = await runExportPipeline({ clips: [clip], outputPath: '/output.mov', codec: 'prores', width: 16, height: 16, fps: 24, quality: 3 }, controller.signal, dependencies(async () => { calls++; controller.abort(); return { success: true } }, async () => { throw new Error('audio pass must not run') }, unlinked))
  assert.deepEqual(result, { success: false, error: 'Export cancelled' })
  assert.equal(calls, 1)
  assert.deepEqual(unlinked, ['/tmp/ltx-export-video-test.mov', '/tmp/ltx-export-audio-test.wav', '/tmp/ltx-pcm-test.raw', '/tmp/ltx-filter-v-test.txt', '/output.ltx-pending-test.mov'])
})
