import assert from 'node:assert/strict'
import test from 'node:test'
import { buildExportVideoPlan } from './export-codec'

test('prores_plan_is_direct_422_and_copy_muxed', () => {
  const plan = buildExportVideoPlan('prores', 3)
  assert.deepEqual(plan.pass1VideoArgs, ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le', '-vendor', 'apl0', '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709'])
  assert.equal(plan.tempExtension, '.mov')
  assert.equal(plan.pass1VideoArgs.includes('libx264'), false)
  assert.equal(plan.pass1VideoArgs.includes('-qscale:v'), false)
  assert.equal(plan.muxer, 'mov')
})

test('codec_quality_validation_preserves_prores_profile_zero', () => {
  assert.equal(buildExportVideoPlan('prores', 0).pass1VideoArgs[3], '0')
  for (const quality of [-1, 4, 1.5]) assert.throws(() => buildExportVideoPlan('prores', quality))
  for (const quality of [-1, 52, 1.5]) assert.throws(() => buildExportVideoPlan('h264', quality))
  for (const quality of [0, -1, Infinity]) assert.throws(() => buildExportVideoPlan('vp9', quality))
})

test('h264_and_vp9_render_directly_to_their_target_codecs', () => {
  assert.deepEqual(buildExportVideoPlan('h264', 18).pass1VideoArgs.slice(0, 2), ['-c:v', 'libx264'])
  assert.deepEqual(buildExportVideoPlan('vp9', 8).pass1VideoArgs.slice(0, 2), ['-c:v', 'libvpx-vp9'])
})
