# Nairobiwood — Roadmap & Pending Enhancements

## System Enhancements

### 1. [Pending Enhancement] Dialogue Script to Neural TTS (.wav) & LatentSync Lip-Sync Pipeline
- [ ] **Dialogue Extraction:** Parse `spoken_text` and `character_id` directly from `project.json` / `storyboard.json`.
- [ ] **Voice Assignment & Synthesis:**
  - Map `character_id` to configured voice profile (e.g., Kokoro-82M / Chatterbox / ElevenLabs / local F5-TTS).
  - Automatically synthesize and stage `~/ComfyFactory/outputs/<project>/shots/<shot_id>/dialogue.wav`.
  - Provide audio recording / manual upload slot in Director Studio for director voiceover.
- [ ] **LatentSync Lip-Sync Integration Gate:**
  - Feed rendered shot video (`shots/<shot_id>/video.mp4`) + `dialogue.wav` into LatentSync 1.6 workflow via `continue_from_repaired_frame(mode="lipsync")`.
  - Allow Director Studio to preview dialogue waveform alongside video before final master cut assembly.

---

## Active Production Status: `midnight_signal`

- [x] **Story & Bible Ingestion:** 3 scenes, 8 shots generated from narrative bible with wardrobe & prop continuity.
- [x] **Master Character Portraits:** Generated for `char_d_leston` (LoRA-anchored) and `char_zuri_vance` (Kontext non-LoRA portrait).
- [x] **Start Keyframes:** All 8 shot keyframes generated on Runpod A100 GPU and approved.
- [ ] **Video Generation (Wan 2.2 I2V):**
  - First shot queued: `s01_sh01`.
  - Status: *Paused awaiting Runpod balance top-up*. (Current balance: `-$0.11`, pod `12j937d5oqqtzt` requires positive balance to start).
