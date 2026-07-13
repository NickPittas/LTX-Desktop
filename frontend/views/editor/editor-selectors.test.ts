import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_TRACKS,
  assetSchema,
  timelineClipSchema,
  timelineSchema,
} from '../../types/project-model.ts'
import { createInitialEditorState } from './editor-state.ts'
import { selectExportClipData } from './editor-selectors.ts'

function exportFixture(primaryPath: string, proxyPath: string) {
  const asset = assetSchema.parse({
    id: 'asset-1',
    type: 'video',
    path: primaryPath,
    proxyPath,
    prompt: '',
    resolution: '1920x1080',
    createdAt: 0,
  })
  const clip = timelineClipSchema.parse({
    id: 'clip-1',
    assetId: asset.id,
    type: 'video',
    startTime: 0,
    duration: 1,
    trimStart: 0,
    trimEnd: 0,
    trackIndex: 0,
    asset: null,
  })
  const timeline = timelineSchema.parse({
    id: 'timeline-1',
    name: 'Timeline',
    createdAt: 0,
    tracks: DEFAULT_TRACKS,
    clips: [clip],
  })
  return createInitialEditorState({
    assets: [asset],
    bins: {},
    timelines: [timeline],
    activeTimelineId: timeline.id,
  })
}

function export_video_uses_primary_mov_and_proxy_audio() {
  const [clip] = selectExportClipData(exportFixture('/project/primary.mov', '/project/primary.proxy.mp4'))

  assert.equal(clip.path, '/project/primary.mov')
  assert.equal(clip.audioPath, '/project/primary.proxy.mp4')
}

function export_exr_directory_uses_documented_proxy_fallback() {
  const [clip] = selectExportClipData(exportFixture('/project/shot_exr', '/project/shot.proxy.mp4'))

  assert.equal(clip.path, '/project/shot.proxy.mp4')
  assert.equal(clip.audioPath, '/project/shot.proxy.mp4')
}

test('export_video_uses_primary_mov_and_proxy_audio', export_video_uses_primary_mov_and_proxy_audio)
test('export_exr_directory_uses_documented_proxy_fallback', export_exr_directory_uses_documented_proxy_fallback)
