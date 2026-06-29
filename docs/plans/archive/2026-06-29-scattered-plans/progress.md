# Progress

## Status
Research complete

## Tasks
- Verified official LTX Ingredients IC-LoRA semantics from HF model card, LTX docs, and official ComfyUI workflow.

## Files Changed
- progress.md

## Notes
- Ingredients uses reference-sheet conditioning. Official workflow loads one reference image, repeats it into frames, and feeds it via `LTXAddVideoICLoRAGuide`; it does not require a user driving video.
