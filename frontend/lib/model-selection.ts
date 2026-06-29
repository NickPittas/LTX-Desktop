import type { components } from '../generated/backend-openapi'

export type ModelSelectionOption = components['schemas']['ModelSelectionOption']
export type ModelSelectionWorkflow = components['schemas']['ModelSelectionOptionsResponse']['workflow']
export type ModelCheckpointID = ModelSelectionOption['id']

export interface GroupedModelOptions {
  section: ModelSelectionOption['section']
  sectionLabel: string
  groups: {
    variantGroup: string
    group: string
    options: ModelSelectionOption[]
  }[]
}

function sectionLabel(section: ModelSelectionOption['section']): string {
  switch (section) {
    case 'full':
      return 'Full'
    case 'kijai':
      return 'Kijai'
    case 'gguf':
      return 'GGUF'
    case 'addons':
      return 'Add-ons'
    default:
      return section
  }
}

export function groupModelOptions(options: ModelSelectionOption[]): GroupedModelOptions[] {
  const bySection = new Map<ModelSelectionOption['section'], Map<string, ModelSelectionOption[]>>()

  for (const option of options) {
    let sectionMap = bySection.get(option.section)
    if (!sectionMap) {
      sectionMap = new Map<string, ModelSelectionOption[]>()
      bySection.set(option.section, sectionMap)
    }
    const key = `${option.variant_group}\0${option.group}`
    let list = sectionMap.get(key)
    if (!list) {
      list = []
      sectionMap.set(key, list)
    }
    list.push(option)
  }

  const result: GroupedModelOptions[] = []
  const sectionOrder: ModelSelectionOption['section'][] = ['full', 'kijai', 'gguf', 'addons']

  for (const section of sectionOrder) {
    const sectionMap = bySection.get(section)
    if (!sectionMap) continue

    const groups: GroupedModelOptions['groups'] = []
    for (const [key, options] of sectionMap.entries()) {
      const [variantGroup, group] = key.split('\0')
      groups.push({ variantGroup: variantGroup ?? '', group: group ?? '', options })
    }

    result.push({ section, sectionLabel: sectionLabel(section), groups })
  }

  return result
}

export function findModelOption(
  options: ModelSelectionOption[],
  id: ModelCheckpointID | null | undefined,
): ModelSelectionOption | undefined {
  if (!id) return undefined
  return options.find((option) => option.id === id)
}

export function isModelOptionSelectable(option: ModelSelectionOption): boolean {
  return !option.disabled_reason
}
