import type { components } from '../generated/backend-openapi'

export type ModelComponentPaths = components['schemas']['ModelComponentPaths']
export type ModelProfilePayload = components['schemas']['ModelProfilePayload']
export type ModelProfilePatchPayload = components['schemas']['ModelProfilePatchPayload']
export type ModelProfilesResponse = components['schemas']['ModelProfilesResponse']
export type ModelProfileValidationResponse = components['schemas']['ModelProfileValidationResponse']
export type ModelProfileActivateResponse = components['schemas']['ModelProfileActivateResponse']
export type ModelProfileProblem = components['schemas']['ModelProfileProblem']
export type AdapterPipeline = components['schemas']['AdapterRecommendationResponse']['pipeline']
export type AdapterStatusResponse = components['schemas']['AdapterStatusResponse']
export type AdapterRecommendationResponse = components['schemas']['AdapterRecommendationResponse']

export type ModelProfileActivationErrorCode =
  | 'MODEL_PROFILE_ACTIVATION_GENERATION_RUNNING'
  | 'MODEL_PROFILE_REQUIRED_ARTIFACTS_MISSING'
  | 'MODEL_PROFILE_CHANGED_DURING_ACTIVATION'
  | 'MODEL_PROFILE_ACTIVATION_ERROR'
  | 'UNKNOWN'
