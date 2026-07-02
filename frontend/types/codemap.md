# frontend/types
## Responsibility
TypeScript data-model layer for the renderer. `project-model.ts` is the single source of truth for the project document (Zod schemas + derived types + V1→V2 migration), `project.ts` is a pure static UI catalog (clip-effect definitions + text presets), `model-profile.ts` re-exports OpenAPI-generated model-profile/adapter types.

## Design Patterns
- **Schema-first in `project-model.ts`:** every entity is a `z.object(...)` and its TS type is `z.infer<typeof ...>`. Const tuple arrays (`generationModeValues`, `assetTypeValues`, etc.) feed `z.enum(...)` so literals and types stay in lockstep. `assetSchema` and `assetTakeSchema` carry an optional `generationElapsedSeconds` (persisted from `use-generation` / `use-ic-lora` completion) so generated video cards can display "Generated in M:SS".
- **Versioned document + explicit migration ladder:** `projectV1Schema` (legacy `asset.bin` string) and `projectV2Schema` (current `bins` map + `asset.binId`) with `migrateProjectV1ToV2`; `migrateProjectData` dispatches on `version` (`undefined` ⇒ V1, `2` ⇒ V2, else throws); `normalizeProject` is the public entry.
- **Co-located default constants** parsed through their own schema: `DEFAULT_SUBTITLE_STYLE`, `DEFAULT_TRACKS` (V1/V2/V3/A1/A2), `DEFAULT_CLIP_TRANSITION`, `DEFAULT_COLOR_CORRECTION`, `DEFAULT_LETTERBOX`, `DEFAULT_EFFECT_MASK`, `DEFAULT_TEXT_STYLE`.
- **Schema `.default(...)` for clip fields** (`speed 1`, `reversed false`, `muted false`, `volume 100`, `flipH/V false`, `opacity 100`, transitions/color-correction) so partially-stored clips normalize on parse.
- **`project.ts` is data-only** (no Zod): `EFFECT_DEFINITIONS` keyed by `EffectType`, `TEXT_PRESETS` array — static catalogs consumed by editor UI.
- **`model-profile.ts` is a façade** over `generated/backend-openapi` `components['schemas']`, exposing only the names the app uses.

## Data & Control Flow
- `project-model.ts` schemas ← localStorage JSON (read by `lib/project-storage.readProject`, written by `writeProject` via `projectSchema.parse`); ← `ProjectContext` mutations (`normalizeProject` runs on every persist); ← `useProjectReferencesMigration` (`projectReferenceSchema`/`projectSchema` to validate legacy records).
- `migrateProjectData` runs on every `readProject`; if `migrated === true` or the id was corrected, `writeProject` re-persists the normalized form.
- `project.ts` catalogs flow into the VideoEditor effects panel and text-overlay presets UI.
- `model-profile.ts` types flow into `hooks/use-model-profiles`, `hooks/use-official-adapters`, and the model-profile settings UI.

## Integration Points
- **`generated/backend-openapi`:** `components['schemas']` consumed by `model-profile.ts` (`ModelComponentPaths`, `ModelProfilePayload`, `ModelProfilePatchPayload`, `ModelProfilesResponse`, `ModelProfileValidationResponse`, `ModelProfileActivateResponse`, `AdapterPipeline`, `AdapterStatusResponse`, `AdapterRecommendationResponse`).
- **`contexts/ProjectContext`:** imports `Project`, `Asset`, `AssetTake`, `ProjectTab`, `createDefaultTimeline`, `normalizeProject`.
- **`contexts/ViewContext`:** imports `ViewType`.
- **`lib/project-storage`:** imports `migrateProjectData`, `projectSchema`, `Project`.
- **`hooks/useProjectReferencesMigration`:** imports `projectReferenceSchema`, `projectSchema`, `Project`.
- **`lib/project-asset-metadata-migration`:** imports `Asset`.
- **`types/project.ts` ← `types/project-model`:** imports `EffectType`, `TextOverlayStyle`.

### Key exports
- **`project-model.ts`:** value arrays (`generationModeValues`, `assetTypeValues`, `timelineClipTypeValues`, `transitionTypeValues`, `trackTypeValues`, `trackKindValues`, `subtitlePositionValues`, `fontWeightValues`, `fontStyleValues`, `textAlignValues`, `effectTypeValues`, `effectMaskShapeValues`, `letterboxAspectRatioValues`, `viewTypeValues`, `projectTabValues`); schemas (`generationParamsSchema`, `assetTakeSchema`, `subtitleStyleSchema`, `trackSchema`, `subtitleClipSchema`, `clipTransitionSchema`, `colorCorrectionSchema`, `letterboxSettingsSchema`, `effectMaskSchema`, `clipEffectSchema`, `textOverlayStyleSchema`, `assetSchema`, `timelineClipSchema`, `timelineSchema`, `assetBinsSchema`, `projectV1Schema`, `projectV2Schema`, `projectSchema`, `projectReferenceSchema`); types (`GenerationParams`, `AssetTake`, `Asset`, `Track`, `SubtitleStyle`, `SubtitleClip`, `TransitionType`, `ClipTransition`, `ColorCorrection`, `LetterboxSettings`, `EffectMask`, `EffectType`, `ClipEffect`, `TextOverlayStyle`, `TimelineClip`, `Timeline`, `AssetBins`, `ProjectV1`, `ProjectV2`, `Project`, `ViewType`, `ProjectTab`); helpers (`createAssetBinId`, `createDefaultTimeline`, `migrateProjectData`, `normalizeProject`).
- **`project.ts`:** `EffectParamDef`, `EffectDefinition`, `EFFECT_DEFINITIONS`, `TextPreset`, `TEXT_PRESETS`.
- **`model-profile.ts`:** re-exported OpenAPI types listed above.
