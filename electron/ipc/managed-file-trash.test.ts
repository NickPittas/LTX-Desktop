import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { trashManagedProjectFiles } from './managed-file-trash.ts'

function withFixture(run: (fixture: { root: string; projectId: string; project: string; generations: string; write: (relativePath: string) => string }) => Promise<void> | void): Promise<void> {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ltx-managed-trash-'))
  const projectId = 'project-1'
  const project = path.join(root, projectId)
  const generations = path.join(root, '.ltx-generations')
  fs.mkdirSync(project)
  fs.mkdirSync(generations)
  const write = (relativePath: string): string => {
    const filePath = path.join(root, relativePath)
    fs.mkdirSync(path.dirname(filePath), { recursive: true })
    fs.writeFileSync(filePath, 'managed')
    return filePath
  }
  return Promise.resolve(run({ root, projectId, project, generations, write })).finally(() => fs.rmSync(root, { recursive: true, force: true }))
}

test('returns an empty success without calling Trash', async () => {
  await withFixture(async ({ root, projectId, generations }) => {
    let calls = 0
    assert.deepEqual(await trashManagedProjectFiles({ projectId, filePaths: [] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async () => { calls += 1 },
    }), { success: true, trashedPaths: [], failedPaths: [] })
    assert.equal(calls, 0)
  })
})

test('returns empty success before checking missing managed directories', async () => {
  let calls = 0
  assert.deepEqual(await trashManagedProjectFiles({ projectId: 'project-1', filePaths: [] }, {
    projectAssetsRoot: path.join(os.tmpdir(), 'ltx-missing-assets-root'),
    generationRoot: path.join(os.tmpdir(), 'ltx-missing-assets-root', '.ltx-generations'),
    trashItem: async () => { calls += 1 },
  }), { success: true, trashedPaths: [], failedPaths: [] })
  assert.equal(calls, 0)
})

test('deduplicates project and generation files after preflight', async () => {
  await withFixture(async ({ root, projectId, generations, write }) => {
    const projectFile = write(`${projectId}/asset.mp4`)
    const generationFile = write('.ltx-generations/source.mp4')
    const calls: string[] = []
    const result = await trashManagedProjectFiles({ projectId, filePaths: [projectFile, generationFile, projectFile] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async filePath => { calls.push(filePath) },
    })
    assert.deepEqual(result, { success: true, trashedPaths: calls, failedPaths: [] })
    assert.equal(calls.length, 2)
  })
})

test('fails preflight without calling Trash for unsafe paths and project IDs', async () => {
  await withFixture(async ({ root, projectId, generations, write }) => {
    const projectFile = write(`${projectId}/asset.mp4`)
    const outsideFile = write('outside.mp4')
    const directory = path.join(root, projectId, 'directory')
    fs.mkdirSync(directory)
    const missing = path.join(root, projectId, 'missing.mp4')
    const cases = [
      { projectId: '', filePaths: [projectFile] },
      { projectId: '../escape', filePaths: [projectFile] },
      { projectId: 'nested/project', filePaths: [projectFile] },
      { projectId, filePaths: ['relative.mp4'] },
      { projectId, filePaths: [path.parse(root).root] },
      { projectId, filePaths: [missing] },
      { projectId, filePaths: [directory] },
      { projectId, filePaths: [outsideFile] },
      { projectId, filePaths: [projectFile, missing] },
    ]
    for (const input of cases) {
      let calls = 0
      const result = await trashManagedProjectFiles(input, {
        projectAssetsRoot: root,
        generationRoot: generations,
        trashItem: async () => { calls += 1 },
      })
      assert.equal(result.success, false)
      assert.equal(calls, 0)
    }
  })
})

test('rejects a symlink that escapes a managed root before calling Trash', async () => {
  await withFixture(async ({ root, projectId, generations, write }) => {
    const outsideFile = write('outside.mp4')
    const escaped = path.join(root, projectId, 'escaped.mp4')
    fs.symlinkSync(outsideFile, escaped)
    let calls = 0
    const result = await trashManagedProjectFiles({ projectId, filePaths: [escaped] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async () => { calls += 1 },
    })
    assert.equal(result.success, false)
    assert.equal(calls, 0)
  })
})

test('rejects an internal candidate symlink before calling Trash', async () => {
  await withFixture(async ({ root, projectId, generations, write }) => {
    const target = write(`${projectId}/target.mp4`)
    const candidate = path.join(root, projectId, 'candidate.mp4')
    fs.symlinkSync(target, candidate)
    let calls = 0
    const result = await trashManagedProjectFiles({ projectId, filePaths: [candidate] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async () => { calls += 1 },
    })
    assert.equal(result.success, false)
    assert.equal(calls, 0)
  })
})

test('rejects an internal directory symlink before calling Trash', async () => {
  await withFixture(async ({ root, projectId, generations, write }) => {
    const targetDirectory = path.join(root, projectId, 'target')
    write(`${projectId}/target/asset.mp4`)
    const candidate = path.join(root, projectId, 'alias', 'asset.mp4')
    fs.symlinkSync(targetDirectory, path.join(root, projectId, 'alias'), 'dir')
    let calls = 0
    const result = await trashManagedProjectFiles({ projectId, filePaths: [candidate] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async () => { calls += 1 },
    })
    assert.equal(result.success, false)
    assert.equal(calls, 0)
  })
})

test('rejects a project-directory symlink to a sibling before calling Trash', async () => {
  await withFixture(async ({ root, projectId, generations, write, project }) => {
    const sibling = path.join(root, 'project-sibling')
    fs.mkdirSync(sibling)
    const candidate = write('project-sibling/asset.mp4')
    fs.rmSync(project, { recursive: true })
    fs.symlinkSync(sibling, project)
    let calls = 0
    const result = await trashManagedProjectFiles({ projectId, filePaths: [candidate] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async () => { calls += 1 },
    })
    assert.equal(result.success, false)
    assert.equal(calls, 0)
  })
})

test('rejects a generation-root symlink to a sibling before calling Trash', async () => {
  await withFixture(async ({ root, projectId, generations, write }) => {
    const sibling = path.join(root, 'generation-sibling')
    fs.mkdirSync(sibling)
    const candidate = write('generation-sibling/asset.mp4')
    fs.rmSync(generations, { recursive: true })
    fs.symlinkSync(sibling, generations)
    let calls = 0
    const result = await trashManagedProjectFiles({ projectId, filePaths: [candidate] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async () => { calls += 1 },
    })
    assert.equal(result.success, false)
    assert.equal(calls, 0)
  })
})

test('reports callback failures after trashing the remaining preflighted files', async () => {
  await withFixture(async ({ root, projectId, generations, write }) => {
    const first = write(`${projectId}/first.mp4`)
    const second = write('.ltx-generations/second.mp4')
    const result = await trashManagedProjectFiles({ projectId, filePaths: [first, second] }, {
      projectAssetsRoot: root,
      generationRoot: generations,
      trashItem: async filePath => {
        if (filePath === fs.realpathSync.native(first)) throw new Error('trash unavailable')
      },
    })
    assert.deepEqual(result, {
      success: true,
      trashedPaths: [fs.realpathSync.native(second)],
      failedPaths: [{ path: fs.realpathSync.native(first), reason: 'trash unavailable' }],
    })
  })
})
