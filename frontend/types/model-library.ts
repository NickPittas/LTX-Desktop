import type { components } from '../generated/backend-openapi'

export type ModelLibraryScanResponse = components['schemas']['ModelLibraryScanResponse']
export type ModelLibraryArtifact = components['schemas']['ModelLibraryArtifact']
export type PartialFile = components['schemas']['PartialFile']
export type UnknownFile = components['schemas']['UnknownFile']

export type ArtifactStatus = ModelLibraryArtifact['status']
export type ArtifactSupportStatus = ModelLibraryArtifact['support_status']
export type ArtifactKind = ModelLibraryArtifact['artifact_kind']
export type CatalogSection = ModelLibraryArtifact['section']
export type ContentPieceId = NonNullable<ModelLibraryArtifact['cp_id']>
export type AdapterId = NonNullable<ModelLibraryArtifact['adapter_id']>

export type ModelDownloadStartResponse = components['schemas']['ModelDownloadStartResponse']
export type ModelDownloadProgressResponse =
  | components['schemas']['DownloadProgressRunningResponse']
  | components['schemas']['DownloadProgressCompleteResponse']
  | components['schemas']['DownloadProgressCancelledResponse']
  | components['schemas']['DownloadProgressErrorResponse']

export type DownloadCancelResponse =
  | components['schemas']['DownloadCancelCancellingResponse']
  | components['schemas']['DownloadCancelNoActiveResponse']
