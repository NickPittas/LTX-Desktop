import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import type { Asset, Project } from '../types/project-model'
import {
  buildManagedDeletionPlan,
  type DeletionTarget,
} from './managed-file-references.ts'

const projectRoot = '/projects/current'
const generationRoot = '/generations'

function asset(id: string, path: string, patch: Partial<Asset> = {}): Asset {
  return {
    id,
    type: 'video',
    path,
    prompt: '',
    resolution: '1280x720',
    createdAt: 1,
    origin: 'generated',
    ...patch,
  }
}

function project(id: string, assets: Asset[], clips: unknown[] = []): Project {
  return {
    version: 2,
    id,
    name: id,
    createdAt: 1,
    updatedAt: 1,
    bins: {},
    assets,
    timelines: [{ id: 'timeline', name: 'Timeline', createdAt: 1, tracks: [], clips: clips as Project['timelines'][number]['clips'], subtitles: [] }],
  }
}

function analyze(currentProject: Project, targets: DeletionTarget[], persistedProjects: Parameters<typeof buildManagedDeletionPlan>[0]['persistedProjects'], patch: Partial<Parameters<typeof buildManagedDeletionPlan>[0]> = {}) {
  return buildManagedDeletionPlan({
    persistedProjects,
    currentProject,
    editorSnapshot: null,
    targets,
    projectManagedDirectory: projectRoot,
    generationRoot,
    ...patch,
  })
}

test('fails closed while legacy project migration is pending', () => {
  const source = readFileSync(new URL('./project-storage.ts', import.meta.url), 'utf8')
  assert.match(source, /localStorage\.getItem\('ltx-projects'\) !== null/)
  assert.match(source, /error: 'Legacy project migration is pending'/)
})

test('deduplicates eligible candidates', () => {
  const current = project('current', [asset('a', `${projectRoot}/a.mp4`, { proxyPath: `${projectRoot}/a.mp4` })])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [`${projectRoot}/a.mp4`] })
})

test('projects a multi-delete as one batch', () => {
  const current = project('current', [asset('a', `${projectRoot}/a.mp4`), asset('b', `${projectRoot}/b.mp4`)])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }, { assetId: 'b' }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [`${projectRoot}/a.mp4`, `${projectRoot}/b.mp4`] })
})

test('keeps shared asset and take paths out of eligibility', () => {
  const shared = `${projectRoot}/shared.mp4`
  const current = project('current', [
    asset('a', shared, { takes: [
      { path: shared, origin: 'generated', createdAt: 1 },
      { path: `${projectRoot}/other.mp4`, origin: 'generated', createdAt: 2 },
    ], activeTakeIndex: 0 }),
    asset('b', shared),
  ])
  assert.deepEqual(analyze(current, [{ assetId: 'a', takeIndex: 0 }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [] })
})

test('repoints the active take alias before counting references', () => {
  const oldTake = `${projectRoot}/old.mp4`
  const nextTake = `${projectRoot}/next.mp4`
  const current = project('current', [asset('a', oldTake, {
    takes: [
      { path: oldTake, origin: 'generated', createdAt: 1 },
      { path: nextTake, origin: 'generated', createdAt: 2 },
    ],
    activeTakeIndex: 0,
  })])
  assert.deepEqual(analyze(current, [{ assetId: 'a', takeIndex: 0 }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [oldTake] })
})

test('counts embedded timeline asset snapshots', () => {
  const path = `${projectRoot}/a.mp4`
  const current = project('current', [asset('a', path)], [{ assetId: null, asset: asset('snapshot', path) }])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [] }), { ok: true, eligiblePaths: [] })
})

test('project-mode embedded clips preserve references', () => {
  const path = `${projectRoot}/a.mp4`
  const current = project('current', [asset('a', path)], [{ assetId: 'a', asset: asset('snapshot', path) }])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [] }), { ok: true, eligiblePaths: [] })
})

test('editor-mode removes deleted clips before counting references', () => {
  const path = `${projectRoot}/a.mp4`
  const current = project('current', [asset('a', path)], [{ assetId: 'a', asset: asset('snapshot', path) }])
  const editorProject = project('current', current.assets, current.timelines[0].clips)
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [] }, {
    editorSnapshot: { projectId: 'current', model: { assets: editorProject.assets, bins: {}, timelines: editorProject.timelines, activeTimelineId: 'timeline' } },
  }), { ok: true, eligiblePaths: [path] })
})

test('uses the unsaved editor overlay and excludes stale persisted current state', () => {
  const path = `${projectRoot}/a.mp4`
  const current = project('current', [asset('a', path), asset('stale-reference', path, { origin: 'unknown' })])
  const editorProject = project('current', [asset('a', path)])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }, {
    editorSnapshot: { projectId: 'current', model: { assets: editorProject.assets, bins: {}, timelines: editorProject.timelines, activeTimelineId: 'timeline' } },
  }), { ok: true, eligiblePaths: [path] })
})

test('fails closed when persisted projects cannot be read', () => {
  const current = project('current', [asset('a', `${projectRoot}/a.mp4`)])
  assert.equal(analyze(current, [{ assetId: 'a' }], { ok: false, error: 'Persisted projects unavailable' }).ok, false)
})

test('rejects an editor snapshot for another project', () => {
  const current = project('current', [asset('a', `${projectRoot}/a.mp4`)])
  const result = analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }, {
    editorSnapshot: { projectId: 'other', model: { assets: [], bins: {}, timelines: [], activeTimelineId: null } },
  })
  assert.deepEqual(result, { ok: false, error: 'Editor snapshot project does not match current project' })
})

test('never creates candidates for historical unknown or imported records', () => {
  const current = project('current', [
    asset('unknown', `${projectRoot}/unknown.mp4`, { origin: 'unknown' }),
    asset('imported', `${projectRoot}/imported.mp4`, { origin: 'imported' }),
  ])
  assert.deepEqual(analyze(current, [{ assetId: 'unknown' }, { assetId: 'imported' }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [] })
})

test('uses generated sibling takes without treating an imported original as a candidate', () => {
  const generatedTake = `${projectRoot}/generated-take.mp4`
  const current = project('current', [asset('imported', `${projectRoot}/imported.mp4`, {
    origin: 'imported',
    takes: [{ path: generatedTake, origin: 'generated', createdAt: 1 }],
  })])
  assert.deepEqual(analyze(current, [{ assetId: 'imported' }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [generatedTake] })
})

test('includes generated source sidecars only under the generation root', () => {
  const source = `${generationRoot}/a.json`
  const current = project('current', [asset('a', `${projectRoot}/a.mp4`, { managedSourcePaths: [source] })])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [source, `${projectRoot}/a.mp4`] })
})

test('enforces exact managed-root containment', () => {
  const current = project('current', [asset('a', '/projects/current-other/a.mp4')])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [] })
})

test('allows paths at an exact managed root and with a trailing root separator', () => {
  const root = '/projects/exact'
  const current = project('current', [asset('a', root), asset('b', `${root}/b.mp4`)])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }, { assetId: 'b' }], { ok: true, projects: [current] }, {
    projectManagedDirectory: `${root}/`,
  }), { ok: true, eligiblePaths: [root, `${root}/b.mp4`] })
})

test('rejects filesystem roots as managed project directories', () => {
  const current = project('current', [asset('a', '/a.mp4')])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }, {
    projectManagedDirectory: '/',
  }), { ok: false, error: 'Project managed directory must not be a filesystem root' })
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }, {
    projectManagedDirectory: 'C:\\',
  }), { ok: false, error: 'Project managed directory must not be a filesystem root' })
})

test('skips malformed candidate paths', () => {
  const current = project('current', [asset('a', 'relative.mp4', { managedSourcePaths: ['../escape.json'] })])
  assert.deepEqual(analyze(current, [{ assetId: 'a' }], { ok: true, projects: [current] }), { ok: true, eligiblePaths: [] })
})
