import { createStudioBridge, type StudioBridge } from './bridge';

if (!Object.hasOwn(HTMLElement.prototype, 'focusGroup')) {
  import('@microsoft/focusgroup-polyfill').then(({ polyfill }) => polyfill(document.documentElement));
}

interface CharacterAsset {
  name: string;
  char_id?: string;
  mode?: string;
  master_portrait_path?: string;
  has_flux_lora?: boolean;
  has_wan22_lora?: boolean;
  lora_files?: string[];
  dataset_images?: number;
}

interface BibleCharacter {
  id: string;
  name: string;
  appearance?: string;
  default_wardrobe?: string;
  default_prop_ids?: string[];
  casting?: {
    master_portrait_path?: string | null;
    nairobiwood?: {
      mode?: string;
      lora_name?: string | null;
      trigger_word?: string;
      lora_strength?: number;
    };
  };
}

interface SceneCharacterState {
  character_id: string;
  wardrobe?: string | null;
  prop_ids?: string[] | null;
  continuity_mode?: string;
  resolved_wardrobe?: string;
  resolved_prop_ids?: string[];
}

interface BibleScene {
  id: string;
  name: string;
  order: number;
  plot_summary?: string;
  dop_direction?: Record<string, string>;
  character_states?: SceneCharacterState[];
  shot_ids?: string[];
}

interface BibleSummary {
  title: string;
  logline: string;
  workflow_stage?: string;
  characters?: BibleCharacter[];
  props?: Array<{ id: string; name: string; description?: string }>;
  scenes?: BibleScene[];
}

interface BranchRecord {
  branch_id: string;
  mode: string;
  branched_preview_path: string;
  branched_master_path: string;
}

interface BranchCheckpoint {
  checkpoint_id: string;
  checkpoint_dir: string;
  frame_index: number;
  seam_ref_frame_path?: string;
  raw_frame_path?: string;
  repaired_frame_path?: string;
  seam_continuity_score?: number;
  branches: BranchRecord[];
}

interface ShotManifest {
  project_name: string;
  shot_id: string;
  scene_id?: string;
  scene_name?: string;
  shot_order?: number;
  character_name: string;
  character_mode?: string;
  prompt: string;
  video_motion_prompt?: string;
  beat_summary?: string;
  spoken_text?: string | null;
  audio_cue?: string;
  inherited_wardrobe?: string;
  inherited_props?: string[];
  duration_sec: number;
  fps: number;
  total_frames: number;
  preview_video_path?: string;
  master_video_path?: string;
  keyframes: {
    status?: string;
    start_frame_path?: string;
    end_frame_path?: string;
    has_end_frame: boolean;
    approved: boolean;
    start_prompt: string;
    end_prompt?: string;
    reference_anchor_path?: string | null;
    last_edit_suggestion?: string | null;
    seed: number;
    lora_strength: number;
  };
  checkpoints: Record<string, BranchCheckpoint>;
}

interface StudioState {
  factory?: { active: boolean; gpu_name?: string; hourly_rate_usd?: number };
  studio_url?: string;
  assets?: {
    characters?: CharacterAsset[];
    datasets?: Array<{ name: string; image_count: number }>;
    loras?: Record<string, string[]>;
    projects?: Array<{ project_name: string; shots: string[]; has_bible?: boolean }>;
  };
  active_project?: string;
  active_shot?: string;
  requested_tab?: 'keyframes' | 'timeline';
  shot_manifest?: ShotManifest;
  bible?: BibleSummary | null;
  storyboard?: ShotManifest[];
}

interface BranchOption {
  label: string;
  videoPath: string;
  ckpt?: BranchCheckpoint;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

const tabs = [$<HTMLButtonElement>('tab-keyframes'), $<HTMLButtonElement>('tab-timeline')];
const panels = [$<HTMLDivElement>('panel-keyframes'), $<HTMLDivElement>('panel-timeline')];

function selectTab(index: number): void {
  tabs.forEach((tab, idx) => {
    const active = idx === index;
    tab.ariaSelected = String(active);
    tab.toggleAttribute('focusgroupstart', active);
    const panel = panels[idx];
    if (active) {
      panel.removeAttribute('hidden');
      panel.inert = false;
    } else {
      panel.setAttribute('hidden', 'until-found');
      panel.inert = true;
    }
  });
}

tabs.forEach((tab, idx) => {
  tab.addEventListener('click', () => selectTab(idx));
  tab.addEventListener('focus', () => selectTab(idx));
});

panels.forEach((panel, idx) => {
  panel.addEventListener('beforematch', () => selectTab(idx));
});

selectTab(0);

function escapeHtml(raw: string): string {
  return raw
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function buildCharacterCatalog(assets?: StudioState['assets']): CharacterAsset[] {
  if (assets?.characters && assets.characters.length > 0) {
    return assets.characters;
  }
  const map = new Map<string, CharacterAsset>();
  for (const [name, files] of Object.entries(assets?.loras ?? {})) {
    map.set(name, {
      name,
      has_flux_lora: files.some((f) => f.toLowerCase().includes('flux')),
      has_wan22_lora: files.some((f) => f.toLowerCase().includes('wan')),
      lora_files: files,
      dataset_images: 0,
    });
  }
  for (const ds of assets?.datasets ?? []) {
    const existing = map.get(ds.name);
    if (existing) {
      existing.dataset_images = ds.image_count;
    } else {
      map.set(ds.name, {
        name: ds.name,
        has_flux_lora: false,
        has_wan22_lora: false,
        lora_files: [],
        dataset_images: ds.image_count,
      });
    }
  }
  return Array.from(map.values()).sort((a, b) => a.name.localeCompare(b.name));
}

function formatCharacterLabel(char: CharacterAsset): string {
  const tags: string[] = [];
  if (char.has_flux_lora && char.has_wan22_lora) {
    tags.push('FLUX + Wan 2.2 LoRA');
  } else if (char.has_flux_lora) {
    tags.push('FLUX LoRA');
  } else if (char.has_wan22_lora) {
    tags.push('Wan 2.2 LoRA');
  } else if (char.master_portrait_path) {
    tags.push('Master Portrait Kontext');
  } else if (char.mode === 'description_only') {
    tags.push('Description Only');
  } else if (char.dataset_images && char.dataset_images > 0) {
    tags.push(`Refs (${char.dataset_images} imgs)`);
  }
  return tags.length > 0 ? `${char.name} (${tags.join(', ')})` : char.name;
}

function populateCharacterDropdown(
  selectEl: HTMLSelectElement,
  characters: CharacterAsset[],
  preferredCharacter?: string,
): void {
  const previousValue = selectEl.value;
  const options: Array<{ value: string; label: string }> = characters.map((c) => ({
    value: c.name,
    label: formatCharacterLabel(c),
  }));

  if (preferredCharacter && !options.some((o) => o.value === preferredCharacter)) {
    options.push({ value: preferredCharacter, label: `${preferredCharacter} (from shot)` });
  }

  if (options.length === 0) {
    selectEl.innerHTML = `<option value="ensemble">ensemble (Description Only)</option>`;
    return;
  }

  selectEl.innerHTML = options
    .map((o) => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`)
    .join('');

  const target = preferredCharacter || previousValue || options[0].value;
  if (options.some((o) => o.value === target)) {
    selectEl.value = target;
  } else {
    selectEl.value = options[0].value;
  }
}

class MaskPainter {
  private ctx: CanvasRenderingContext2D;
  private drawing = false;
  public eraseMode = false;
  public brushSize = 24;
  public hasStrokes = false;

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas unsupported');
    this.ctx = ctx;

    canvas.addEventListener('pointerdown', (e) => {
      this.drawing = true;
      canvas.setPointerCapture(e.pointerId);
      this.paintAt(e);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (this.drawing) this.paintAt(e);
    });
    canvas.addEventListener('pointerup', () => {
      this.drawing = false;
    });
  }

  resize(width: number, height: number): void {
    if (width > 0 && height > 0) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.hasStrokes = false;
  }

  private paintAt(e: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / Math.max(rect.width, 1)) * this.canvas.width;
    const y = ((e.clientY - rect.top) / Math.max(rect.height, 1)) * this.canvas.height;
    this.ctx.save();
    this.ctx.globalCompositeOperation = this.eraseMode ? 'destination-out' : 'source-over';
    this.ctx.fillStyle = 'rgba(255, 64, 64, 0.72)';
    this.ctx.beginPath();
    this.ctx.arc(x, y, this.brushSize, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.restore();
    if (!this.eraseMode) this.hasStrokes = true;
  }

  exportBinaryMaskDataUrl(): string {
    const off = document.createElement('canvas');
    off.width = this.canvas.width || 1280;
    off.height = this.canvas.height || 720;
    const octx = off.getContext('2d')!;
    octx.fillStyle = '#000000';
    octx.fillRect(0, 0, off.width, off.height);
    const src = this.ctx.getImageData(0, 0, off.width, off.height);
    const dst = octx.getImageData(0, 0, off.width, off.height);
    for (let i = 0; i < src.data.length; i += 4) {
      const val = src.data[i + 3] > 10 ? 255 : 0;
      dst.data[i] = val;
      dst.data[i + 1] = val;
      dst.data[i + 2] = val;
      dst.data[i + 3] = 255;
    }
    octx.putImageData(dst, 0, 0);
    return off.toDataURL('image/png');
  }
}

async function init(): Promise<void> {
  const statusBanner = $<HTMLDivElement>('status-banner');
  const setStatus = (msg: string) => {
    statusBanner.textContent = msg;
  };

  let currentState: StudioState = {};
  let studioUrl = 'http://127.0.0.1:8190';
  let activeVideoPath = '';
  let currentFps = 24;
  let currentTotalFrames = 81;
  let activeCheckpointDir = '';
  let activeMaskTarget: 'start' | 'end' = 'start';
  let branchOptions: BranchOption[] = [];
  let loadedShotKey = '';
  let selectedSceneFilter = '';

  const timelinePainter = new MaskPainter($<HTMLCanvasElement>('timeline-mask-canvas'));
  const dialogPainter = new MaskPainter($<HTMLCanvasElement>('dialog-mask-canvas'));

  $<HTMLInputElement>('kf-lora').addEventListener('input', (e) => {
    $<HTMLOutputElement>('kf-lora-out').value = (e.target as HTMLInputElement).value;
  });
  $<HTMLInputElement>('rep-denoise').addEventListener('input', (e) => {
    $<HTMLOutputElement>('rep-denoise-out').value = (e.target as HTMLInputElement).value;
  });
  $<HTMLInputElement>('dialog-brush-size').addEventListener('input', (e) => {
    dialogPainter.brushSize = Number((e.target as HTMLInputElement).value);
  });
  $<HTMLInputElement>('brush-enable').addEventListener('change', (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    $<HTMLCanvasElement>('timeline-mask-canvas').classList.toggle('drawing-active', enabled);
  });
  $<HTMLButtonElement>('btn-brush-mode').addEventListener('click', (e) => {
    timelinePainter.eraseMode = !timelinePainter.eraseMode;
    (e.target as HTMLButtonElement).textContent = timelinePainter.eraseMode ? 'Mode: Erase' : 'Mode: Brush';
  });
  $<HTMLInputElement>('brush-size').addEventListener('input', (e) => {
    timelinePainter.brushSize = Number((e.target as HTMLInputElement).value);
  });
  $<HTMLButtonElement>('btn-brush-clear').addEventListener('click', () => timelinePainter.clear());
  $<HTMLButtonElement>('btn-dialog-clear').addEventListener('click', () => dialogPainter.clear());

  const maskDialog = $<HTMLDialogElement>('mask-dialog');
  $<HTMLButtonElement>('btn-mask-dialog-close').addEventListener('click', () => maskDialog.close('cancel'));

  const bridge: StudioBridge = await createStudioBridge();

  $<HTMLButtonElement>('btn-open-browser').addEventListener('click', () => {
    void bridge.openExternalUrl(studioUrl);
  });

  maskDialog.addEventListener('close', async () => {
    const val = maskDialog.returnValue;
    maskDialog.returnValue = '';
    if (val !== 'apply' || !currentState.active_project || !currentState.active_shot) return;
    try {
      const suggestion = $<HTMLInputElement>('dialog-prompt').value.trim() || undefined;
      const targetText = $<HTMLInputElement>('dialog-target-text').value.trim() || undefined;
      let maskPath: string | undefined;
      if (dialogPainter.hasStrokes) {
        setStatus(`Saving painted area mask for ${activeMaskTarget} keyframe...`);
        const maskRes = await bridge.callTool<{ mask_image_path: string }>('app_save_painted_mask', {
          project_name: currentState.active_project,
          shot_id: currentState.active_shot,
          mask_data_url: dialogPainter.exportBinaryMaskDataUrl(),
          label: `kf_${activeMaskTarget}`,
        });
        maskPath = maskRes.mask_image_path;
      }
      setStatus(`Rebuilding selected area on ${activeMaskTarget} keyframe...`);
      const updated = await bridge.callTool<StudioState>('regenerate_shot_keyframe', {
        project_name: currentState.active_project,
        shot_id: currentState.active_shot,
        which: activeMaskTarget,
        edit_suggestion: suggestion,
        mask_image_path: maskPath,
        mask_target_text: targetText,
        repair_engine: $<HTMLSelectElement>('dialog-engine').value,
      });
      await applyState(updated, { forceFormSync: true });
      setStatus(`Keyframe (${activeMaskTarget}) rebuilt and ready for review.`);
    } catch (err) {
      setStatus(`Keyframe area rebuild failed: ${(err as Error).message}`);
    }
  });

  async function readJsonFromFactory<T>(relPath: string): Promise<T | null> {
    try {
      const url = await bridge.resolveMediaUrl(relPath);
      if (!url) return null;
      if (url.startsWith('data:')) {
        const commaIdx = url.indexOf(',');
        if (commaIdx < 0) return null;
        const b64 = url.slice(commaIdx + 1);
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const text = new TextDecoder('utf-8').decode(bytes);
        return JSON.parse(text) as T;
      }
      const resp = await fetch(url);
      if (!resp.ok) return null;
      return (await resp.json()) as T;
    } catch {
      return null;
    }
  }

  async function hydrateBibleAndStoryboardIfMissing(state: StudioState): Promise<void> {
    const proj = state.active_project || state.assets?.projects?.[0]?.project_name;
    if (!proj) return;
    state.active_project = proj;

    if (!state.bible) {
      const rawBible = await readJsonFromFactory<{
        project?: { title?: string; logline?: string; workflow_stage?: string };
        assets?: {
          characters?: BibleCharacter[];
          props?: Array<{ id: string; name: string; description?: string }>;
        };
        scenes?: Array<{
          id: string;
          name: string;
          order?: number;
          plot_summary?: string;
          dop_direction?: Record<string, string>;
          character_states?: SceneCharacterState[];
          shots?: Array<{ id: string }>;
        }>;
      }>(`outputs/${proj}/project.json`);

      if (rawBible) {
        state.bible = {
          title: rawBible.project?.title || proj,
          logline: rawBible.project?.logline || '',
          workflow_stage: rawBible.project?.workflow_stage || 'storyboarding',
          characters: rawBible.assets?.characters ?? [],
          props: rawBible.assets?.props ?? [],
          scenes: (rawBible.scenes ?? []).map((sc, idx) => ({
            id: sc.id,
            name: sc.name,
            order: sc.order ?? idx + 1,
            plot_summary: sc.plot_summary ?? '',
            dop_direction: sc.dop_direction ?? {},
            character_states: sc.character_states ?? [],
            shot_ids: (sc.shots ?? []).map((sh) => sh.id),
          })),
        };
      }
    }

    const matchingProj = state.assets?.projects?.find((p) => p.project_name === proj);
    let shotIds: string[] = [];
    if (state.bible?.scenes && state.bible.scenes.length > 0) {
      for (const sc of state.bible.scenes) {
        for (const sid of sc.shot_ids ?? []) {
          if (!shotIds.includes(sid)) shotIds.push(sid);
        }
      }
    }
    for (const sid of matchingProj?.shots ?? []) {
      if (!shotIds.includes(sid)) shotIds.push(sid);
    }

    if ((!state.storyboard || state.storyboard.length === 0) && shotIds.length > 0) {
      const loadedShots: ShotManifest[] = [];
      for (const sid of shotIds) {
        const sm = await readJsonFromFactory<ShotManifest>(`outputs/${proj}/shots/${sid}/shot.manifest.json`);
        if (sm) loadedShots.push(sm);
      }
      if (loadedShots.length > 0) {
        state.storyboard = loadedShots;
      }
    }

    if (state.active_shot && state.storyboard) {
      const fullActive = state.storyboard.find((s) => s.shot_id === state.active_shot);
      if (fullActive) {
        state.shot_manifest = {
          ...fullActive,
          ...state.shot_manifest,
          video_motion_prompt: fullActive.video_motion_prompt || state.shot_manifest?.video_motion_prompt,
          audio_cue: fullActive.audio_cue || state.shot_manifest?.audio_cue,
          spoken_text: fullActive.spoken_text ?? state.shot_manifest?.spoken_text,
          inherited_wardrobe: fullActive.inherited_wardrobe || state.shot_manifest?.inherited_wardrobe,
          inherited_props: fullActive.inherited_props ?? state.shot_manifest?.inherited_props,
          scene_id: fullActive.scene_id || state.shot_manifest?.scene_id,
          scene_name: fullActive.scene_name || state.shot_manifest?.scene_name,
          character_mode: fullActive.character_mode || state.shot_manifest?.character_mode,
        };
      }
    }
  }

  async function applyImage(imgId: string, emptyId: string | null, localPath?: string): Promise<void> {
    const img = $<HTMLImageElement>(imgId);
    const empty = emptyId ? $<HTMLElement>(emptyId) : null;
    if (!localPath) {
      img.classList.remove('visible');
      img.removeAttribute('src');
      if (empty) empty.hidden = false;
      return;
    }
    try {
      const url = await bridge.resolveMediaUrl(localPath);
      img.src = url;
      img.classList.add('visible');
      if (empty) empty.hidden = true;
    } catch {
      img.classList.remove('visible');
      if (empty) empty.hidden = false;
    }
  }

  async function renderCharacterStrip(state: StudioState): Promise<void> {
    const strip = $<HTMLDivElement>('character-casting-strip');
    const bibleChars = state.bible?.characters ?? [];
    if (bibleChars.length === 0) {
      strip.innerHTML = '';
      return;
    }

    strip.innerHTML = bibleChars
      .map((ch) => {
        const nw = ch.casting?.nairobiwood ?? {};
        const mode = nw.mode || (ch.casting?.master_portrait_path ? 'kontext_portrait' : 'description_only');
        const modeLabel =
          mode === 'lora'
            ? `LoRA (${nw.lora_name || ch.name})`
            : mode === 'kontext_portrait'
              ? 'Master Portrait Kontext'
              : 'Description Only';
        const badgeClass = mode === 'lora' || ch.casting?.master_portrait_path ? 'badge-ok' : 'badge-warn';
        const thumbId = `char-thumb-${escapeHtml(ch.id)}`;
        return `
          <div class="char-card">
            <img id="${thumbId}" class="char-portrait-thumb" alt="${escapeHtml(ch.name)} master portrait" />
            <div class="char-meta">
              <h3>
                <span>${escapeHtml(ch.name)}</span>
                <span class="badge ${badgeClass}">${escapeHtml(modeLabel)}</span>
              </h3>
              <p title="${escapeHtml(ch.default_wardrobe || ch.appearance || '')}">
                <strong>Wardrobe:</strong> ${escapeHtml(ch.default_wardrobe || ch.appearance || 'Unspecified')}
              </p>
              <button
                type="button"
                class="btn-secondary btn-gen-portrait"
                data-char-id="${escapeHtml(ch.id)}"
                data-char-name="${escapeHtml(ch.name)}"
              >
                ${ch.casting?.master_portrait_path ? 'Update Master Portrait' : 'Generate Master Portrait'}
              </button>
            </div>
          </div>
        `;
      })
      .join('');

    for (const ch of bibleChars) {
      const mp = ch.casting?.master_portrait_path;
      if (mp) {
        const imgEl = document.getElementById(`char-thumb-${ch.id}`) as HTMLImageElement | null;
        if (imgEl) {
          void bridge.resolveMediaUrl(mp).then((url) => {
            if (url) imgEl.src = url;
          });
        }
      }
    }

    strip.querySelectorAll<HTMLButtonElement>('.btn-gen-portrait').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const cid = btn.dataset.charId;
        const cname = btn.dataset.charName || cid;
        const proj = currentState.active_project || $<HTMLInputElement>('bible-project-name').value.trim();
        if (!cid || !proj) return;
        btn.disabled = true;
        try {
          setStatus(`Generating Master Character Portrait for '${cname}'...`);
          const nextState = await bridge.callTool<StudioState>('generate_character_portrait', {
            project_name: proj,
            char_id: cid,
          });
          await applyState(nextState);
          setStatus(`Master Character Portrait ready for '${cname}'.`);
        } catch (err) {
          setStatus(`Master Portrait generation failed: ${(err as Error).message}`);
        } finally {
          btn.disabled = false;
        }
      });
    });
  }

  async function renderStoryboardGrid(state: StudioState): Promise<void> {
    const container = $<HTMLDivElement>('storyboard-container');
    const storyboard = state.storyboard ?? [];
    if (storyboard.length === 0) {
      container.innerHTML = `<p class="muted">No storyboard shots staged yet. Click "Load &amp; Stage Storyboard" above or generate a shot below.</p>`;
      return;
    }

    const scenesMap = new Map<string, { sceneId: string; sceneName: string; shots: ShotManifest[]; states: SceneCharacterState[] }>();
    const bibleScenes = new Map<string, BibleScene>();
    for (const sc of state.bible?.scenes ?? []) {
      bibleScenes.set(sc.id, sc);
    }

    for (const sh of storyboard) {
      const sid = sh.scene_id || 'scene_01';
      if (selectedSceneFilter && sid !== selectedSceneFilter) continue;
      const bScene = bibleScenes.get(sid);
      if (!scenesMap.has(sid)) {
        scenesMap.set(sid, {
          sceneId: sid,
          sceneName: sh.scene_name || bScene?.name || sid,
          shots: [],
          states: bScene?.character_states ?? [],
        });
      }
      scenesMap.get(sid)!.shots.push(sh);
    }

    const propsLookup = new Map<string, string>();
    for (const p of state.bible?.props ?? []) {
      propsLookup.set(p.id, p.name);
    }
    const charsLookup = new Map<string, string>();
    for (const c of state.bible?.characters ?? []) {
      charsLookup.set(c.id, c.name);
    }

    container.innerHTML = Array.from(scenesMap.values())
      .map((group) => {
        const continuityHtml = group.states
          .map((cs) => {
            const cname = charsLookup.get(cs.character_id) || cs.character_id;
            const modeTag = cs.continuity_mode === 'override' ? 'Scene Override' : 'Persisted from Prior Scene';
            const propNames = (cs.resolved_prop_ids ?? []).map((pid) => propsLookup.get(pid) || pid).join(', ');
            return `<span class="continuity-pill"><strong>${escapeHtml(cname)} (${modeTag}):</strong> ${escapeHtml(
              (cs.resolved_wardrobe || '').slice(0, 70),
            )}${propNames ? ` | Props: ${escapeHtml(propNames)}` : ''}</span>`;
          })
          .join('');

        const cardsHtml = group.shots
          .map((sh) => {
            const isActive = sh.shot_id === state.active_shot;
            const hasVideo = Boolean(sh.preview_video_path);
            const hasFrame = Boolean(sh.keyframes.start_frame_path);
            const statusText = hasVideo
              ? 'Video Rendered'
              : sh.keyframes.approved
                ? 'Frame Approved'
                : hasFrame
                  ? 'First Frame Ready'
                  : 'Staged';
            const badgeClass = hasVideo || sh.keyframes.approved ? 'badge-ok' : hasFrame ? 'badge-warn' : 'badge-idle';
            const propsStr = (sh.inherited_props ?? []).join(', ');
            return `
              <article
                class="shot-card ${isActive ? 'active-shot-card' : ''}"
                data-shot-id="${escapeHtml(sh.shot_id)}"
              >
                <div class="card-head">
                  <strong>${escapeHtml(sh.shot_id)}</strong>
                  <span class="badge ${badgeClass}">${escapeHtml(statusText)}</span>
                </div>
                <div class="shot-card-thumb">
                  <img id="sb-thumb-${escapeHtml(sh.shot_id)}" alt="${escapeHtml(sh.shot_id)} first frame" ${
                    hasFrame ? '' : 'hidden'
                  } />
                  ${hasFrame ? '' : `<span class="muted">Staged — Ready to Generate</span>`}
                </div>
                <div class="shot-card-meta">
                  <div><strong>Actor:</strong> ${escapeHtml(sh.character_name)} (${escapeHtml(sh.character_mode || 'portrait')})</div>
                  ${
                    sh.inherited_wardrobe
                      ? `<div title="${escapeHtml(sh.inherited_wardrobe)}"><strong>Wardrobe:</strong> ${escapeHtml(
                          sh.inherited_wardrobe.slice(0, 56),
                        )}...</div>`
                      : ''
                  }
                  ${propsStr ? `<div><strong>Props:</strong> ${escapeHtml(propsStr)}</div>` : ''}
                  ${
                    sh.spoken_text
                      ? `<div><strong>Line:</strong> "${escapeHtml(sh.spoken_text.slice(0, 65))}"</div>`
                      : sh.beat_summary
                        ? `<div>${escapeHtml(sh.beat_summary.slice(0, 65))}</div>`
                        : ''
                  }
                </div>
              </article>
            `;
          })
          .join('');

        return `
          <div class="scene-block">
            <div class="scene-block-head">
              <div>
                <h3>${escapeHtml(group.sceneId.toUpperCase())} — ${escapeHtml(group.sceneName)}</h3>
                ${continuityHtml ? `<div class="scene-continuity-pills">${continuityHtml}</div>` : ''}
              </div>
              <button
                type="button"
                class="btn-secondary btn-gen-scene"
                data-scene-id="${escapeHtml(group.sceneId)}"
              >
                Generate ${escapeHtml(group.sceneId)} First Frames (${group.shots.length})
              </button>
            </div>
            <div class="storyboard-grid">${cardsHtml}</div>
          </div>
        `;
      })
      .join('');

    // Resolve thumbnails asynchronously
    for (const sh of storyboard) {
      if (sh.keyframes.start_frame_path) {
        const imgEl = document.getElementById(`sb-thumb-${sh.shot_id}`) as HTMLImageElement | null;
        if (imgEl) {
          void bridge.resolveMediaUrl(sh.keyframes.start_frame_path).then((url) => {
            if (url) {
              imgEl.src = url;
              imgEl.hidden = false;
            }
          });
        }
      }
    }

    // Wire shot card selection
    container.querySelectorAll<HTMLElement>('.shot-card').forEach((card) => {
      card.addEventListener('click', async () => {
        const sid = card.dataset.shotId;
        const proj = currentState.active_project;
        if (!sid || !proj) return;
        const nextState = await bridge.callTool<StudioState>('app_get_studio_state', {
          project_name: proj,
          shot_id: sid,
        });
        await applyState(nextState, { forceFormSync: true });
        setStatus(`Inspecting shot '${proj}/${sid}'.`);
      });
    });

    // Wire scene-level progressive generation
    container.querySelectorAll<HTMLButtonElement>('.btn-gen-scene').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const sceneId = btn.dataset.sceneId;
        if (!sceneId) return;
        await runProgressiveKeyframes(sceneId);
      });
    });
  }

  async function runProgressiveKeyframes(sceneId?: string): Promise<void> {
    const proj = currentState.active_project || $<HTMLInputElement>('kf-project').value.trim();
    if (!proj) return;
    const shots = (currentState.storyboard ?? []).filter(
      (sh) => !sceneId || sh.scene_id === sceneId,
    );
    if (shots.length === 0) {
      setStatus('No staged shots found for selection.');
      return;
    }

    for (let i = 0; i < shots.length; i++) {
      const sh = shots[i];
      setStatus(`Generating First Frame (${i + 1}/${shots.length}): ${sh.shot_id} (${sh.scene_name || sh.scene_id})...`);
      try {
        const nextState = await bridge.callTool<StudioState>('generate_shot_keyframes', {
          project_name: proj,
          shot_id: sh.shot_id,
          character_name: sh.character_name,
          start_prompt: sh.keyframes.start_prompt,
          end_prompt: sh.keyframes.has_end_frame ? sh.keyframes.end_prompt || undefined : undefined,
          lora_strength: sh.keyframes.lora_strength,
          seed: sh.keyframes.seed,
        });
        await applyState(nextState, { forceFormSync: true });
      } catch (err) {
        setStatus(`Stopped at ${sh.shot_id}: ${(err as Error).message}`);
        return;
      }
    }
    setStatus(`Completed generating ${shots.length} first frame(s)! Inspect and approve before video rendering.`);
  }

  async function applyState(
    state: StudioState,
    opts: { forceFormSync?: boolean } = {},
  ): Promise<void> {
    await hydrateBibleAndStoryboardIfMissing(state);
    currentState = state;
    const badge = $<HTMLSpanElement>('factory-badge');
    if (state.factory?.active) {
      badge.textContent = `RunPod: ${state.factory.gpu_name ?? 'GPU'} ($${state.factory.hourly_rate_usd?.toFixed(2) ?? '?'}/hr)`;
      badge.className = 'badge badge-ok';
    } else {
      badge.textContent = 'RunPod: Idle';
      badge.className = 'badge badge-idle';
    }

    if (state.studio_url) {
      studioUrl = state.studio_url;
    }

    if (state.bible) {
      $<HTMLParagraphElement>('bible-subtitle').textContent =
        `${state.bible.title} — ${state.bible.logline}`;
    }

    await renderCharacterStrip(state);
    await renderStoryboardGrid(state);

    const characters = buildCharacterCatalog(state.assets);
    const manifestChar = state.shot_manifest?.character_name;
    populateCharacterDropdown($<HTMLSelectElement>('kf-character'), characters, manifestChar);
    populateCharacterDropdown($<HTMLSelectElement>('rep-character'), characters, manifestChar);

    const projSelect = $<HTMLSelectElement>('project-select');
    const sceneSelect = $<HTMLSelectElement>('scene-select');
    const shotSelect = $<HTMLSelectElement>('shot-select');
    const projects = state.assets?.projects ?? [];

    if (projects.length === 0) {
      projSelect.innerHTML = `<option value="">(New project)</option>`;
      shotSelect.innerHTML = `<option value="">(New shot)</option>`;
    } else {
      projSelect.innerHTML = projects
        .map((p) => `<option value="${escapeHtml(p.project_name)}">${escapeHtml(p.project_name)}</option>`)
        .join('');
      if (state.active_project) {
        projSelect.value = state.active_project;
        $<HTMLInputElement>('bible-project-name').value = state.active_project;
      }

      const scenes = state.bible?.scenes ?? [];
      sceneSelect.innerHTML =
        `<option value="">All Scenes (${scenes.length})</option>` +
        scenes
          .map((sc) => `<option value="${escapeHtml(sc.id)}">${escapeHtml(sc.id)} — ${escapeHtml(sc.name)}</option>`)
          .join('');
      sceneSelect.value = selectedSceneFilter;

      const matchingProj = projects.find((p) => p.project_name === (state.active_project || projSelect.value));
      const shots = matchingProj?.shots ?? [];
      shotSelect.innerHTML = shots.length > 0
        ? shots.map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join('')
        : `<option value="">(No shots yet)</option>`;
      if (state.active_shot) shotSelect.value = state.active_shot;
    }

    const manifest = state.shot_manifest;
    if (manifest) {
      const nextShotKey = `${manifest.project_name}/${manifest.shot_id}`;
      const shotChanged = nextShotKey !== loadedShotKey;
      loadedShotKey = nextShotKey;

      const syncIfIdle = <T extends HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(
        el: T,
        nextVal: string,
      ) => {
        if (opts.forceFormSync || shotChanged || document.activeElement !== el) {
          el.value = nextVal;
        }
      };

      syncIfIdle($<HTMLInputElement>('kf-project'), manifest.project_name);
      syncIfIdle($<HTMLInputElement>('kf-shot'), manifest.shot_id);
      syncIfIdle($<HTMLSelectElement>('kf-character'), manifest.character_name);
      syncIfIdle($<HTMLSelectElement>('rep-character'), manifest.character_name);
      syncIfIdle($<HTMLTextAreaElement>('kf-start-prompt'), manifest.keyframes.start_prompt);
      syncIfIdle($<HTMLTextAreaElement>('kf-end-prompt'), manifest.keyframes.end_prompt ?? '');
      $<HTMLInputElement>('kf-use-end').checked = manifest.keyframes.has_end_frame;
      syncIfIdle($<HTMLInputElement>('kf-seed'), String(manifest.keyframes.seed));
      $<HTMLInputElement>('kf-lora').value = String(manifest.keyframes.lora_strength);
      $<HTMLOutputElement>('kf-lora-out').value = String(manifest.keyframes.lora_strength);

      // Sync decoupled Video Motion Prompt & Audio Cue in Section 4
      syncIfIdle(
        $<HTMLTextAreaElement>('render-motion-prompt'),
        manifest.video_motion_prompt || manifest.prompt || manifest.keyframes.start_prompt,
      );
      syncIfIdle(
        $<HTMLInputElement>('render-audio-cue'),
        manifest.audio_cue || manifest.spoken_text || '',
      );
      syncIfIdle($<HTMLInputElement>('render-duration'), String(manifest.duration_sec || 6));

      // Populate Continuity Context Banner for the active shot
      const banner = $<HTMLDivElement>('shot-continuity-banner');
      if (manifest.scene_name || manifest.inherited_wardrobe || (manifest.inherited_props ?? []).length > 0) {
        banner.hidden = false;
        const propText = (manifest.inherited_props ?? []).join(', ') || 'None';
        banner.innerHTML = `
          <div><strong>Scene:</strong> ${escapeHtml(manifest.scene_id || '')} — ${escapeHtml(manifest.scene_name || '')}</div>
          <div><strong>Persisted Character Wardrobe:</strong> ${escapeHtml(manifest.inherited_wardrobe || 'Default')}</div>
          <div><strong>Persisted Scene Props:</strong> ${escapeHtml(propText)}</div>
          ${manifest.spoken_text ? `<div><strong>Spoken Beat:</strong> "${escapeHtml(manifest.spoken_text)}"</div>` : ''}
        `;
      } else {
        banner.hidden = true;
      }

      const pill = $<HTMLSpanElement>('kf-approval-pill');
      const hasStart = Boolean(manifest.keyframes.start_frame_path);
      if (manifest.preview_video_path) {
        pill.textContent = 'Approved & Video Rendered';
        pill.className = 'badge badge-ok';
      } else if (manifest.keyframes.approved) {
        pill.textContent = 'First Frame Approved';
        pill.className = 'badge badge-ok';
      } else if (hasStart) {
        pill.textContent = 'First Frame Ready — Awaiting Director Validation';
        pill.className = 'badge badge-warn';
      } else {
        pill.textContent = 'Staged in Storyboard';
        pill.className = 'badge badge-idle';
      }

      await applyImage('img-start-kf', 'empty-start-kf', manifest.keyframes.start_frame_path);
      await applyImage('img-end-kf', 'empty-end-kf', manifest.keyframes.end_frame_path);

      const branchSelect = $<HTMLSelectElement>('branch-select');
      const ckpts = Object.values(manifest.checkpoints ?? {});
      branchOptions = [];
      if (manifest.preview_video_path) {
        branchOptions.push({ label: 'Root Render', videoPath: manifest.preview_video_path });
      }
      for (const ck of ckpts) {
        for (const br of ck.branches) {
          branchOptions.push({
            label: `${ck.checkpoint_id} -> ${br.branch_id} (${br.mode})`,
            videoPath: br.branched_preview_path,
            ckpt: ck,
          });
        }
      }
      branchSelect.innerHTML = branchOptions.length > 0
        ? branchOptions.map((b, i) => `<option value="${i}">${escapeHtml(b.label)}</option>`).join('')
        : `<option value="">(No rendered video)</option>`;

      currentFps = manifest.fps || 24;
      currentTotalFrames = Math.max(1, manifest.total_frames || 81);
      $<HTMLSpanElement>('frame-total-label').textContent = String(currentTotalFrames - 1);
      $<HTMLInputElement>('frame-slider').max = String(currentTotalFrames - 1);

      if (manifest.preview_video_path) {
        activeVideoPath = manifest.preview_video_path;
        await loadTimelineVideo(activeVideoPath);
      }

      if (ckpts.length > 0) {
        const latest = ckpts[ckpts.length - 1];
        activeCheckpointDir = latest.checkpoint_dir;
        $<HTMLSpanElement>('seam-score-badge').textContent =
          `Seam Score: ${latest.seam_continuity_score ?? '—'}`;
        await applyImage('img-seam-prev', null, latest.seam_ref_frame_path);
        await applyImage('img-seam-raw', null, latest.raw_frame_path);
        await applyImage('img-seam-repaired', null, latest.repaired_frame_path);
      }
    }

    if (state.requested_tab === 'timeline') selectTab(1);
    else if (state.requested_tab === 'keyframes') selectTab(0);
  }

  async function loadTimelineVideo(videoPath: string): Promise<void> {
    const videoEl = $<HTMLVideoElement>('timeline-video');
    const frameImg = $<HTMLImageElement>('timeline-frame-img');
    if (bridge.mode === 'browser') {
      videoEl.hidden = false;
      frameImg.hidden = true;
      videoEl.src = await bridge.resolveMediaUrl(videoPath);
      videoEl.onloadedmetadata = () => {
        timelinePainter.resize(videoEl.videoWidth || 1280, videoEl.videoHeight || 720);
      };
    } else {
      videoEl.hidden = true;
      frameImg.hidden = false;
      await scrubToFrame(0);
    }
  }

  async function scrubToFrame(frameIdx: number): Promise<void> {
    const clamped = Math.max(0, Math.min(frameIdx, currentTotalFrames - 1));
    $<HTMLInputElement>('frame-slider').value = String(clamped);
    $<HTMLElement>('frame-index-label').textContent = String(clamped);

    const videoEl = $<HTMLVideoElement>('timeline-video');
    const frameImg = $<HTMLImageElement>('timeline-frame-img');
    if (bridge.mode === 'browser' && !videoEl.hidden) {
      videoEl.pause();
      videoEl.currentTime = clamped / Math.max(currentFps, 1);
      return;
    }
    if (activeVideoPath) {
      const res = await bridge.callTool<{ data_url: string; total_frames?: number }>('app_extract_video_frame', {
        video_path: activeVideoPath,
        frame_index: clamped,
      });
      frameImg.src = res.data_url;
      frameImg.onload = () => {
        timelinePainter.resize(frameImg.naturalWidth || 1280, frameImg.naturalHeight || 720);
      };
    }
  }

  bridge.onStatePush((payload) => {
    void applyState(payload as unknown as StudioState);
  });

  // Production Bible Import form
  $<HTMLFormElement>('form-import-bible').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $<HTMLButtonElement>('btn-import-bible');
    btn.disabled = true;
    try {
      const projName = $<HTMLInputElement>('bible-project-name').value.trim();
      const srcPath = $<HTMLInputElement>('bible-source-path').value.trim();
      setStatus(`Importing & staging Storyboard into ~/ComfyFactory/outputs/${projName}...`);
      const state = await bridge.callTool<StudioState>('import_or_init_story_bible', {
        project_name: projName,
        story_or_bible_path: srcPath || undefined,
      });
      await applyState(state, { forceFormSync: true });
      setStatus(`Staged Storyboard for '${projName}' (${state.storyboard?.length ?? 0} shots across scenes).`);
    } catch (err) {
      setStatus(`Bible import failed: ${(err as Error).message}`);
    } finally {
      btn.disabled = false;
    }
  });

  $<HTMLButtonElement>('btn-gen-active-scene').addEventListener('click', () => {
    const activeScene = selectedSceneFilter || currentState.shot_manifest?.scene_id || 'scene_01';
    void runProgressiveKeyframes(activeScene);
  });

  $<HTMLButtonElement>('btn-gen-all-scenes').addEventListener('click', () => {
    void runProgressiveKeyframes(undefined);
  });

  // Header Project / Scene / Shot selectors
  $<HTMLSelectElement>('project-select').addEventListener('change', async (e) => {
    const proj = (e.target as HTMLSelectElement).value;
    if (!proj) return;
    selectedSceneFilter = '';
    const state = await bridge.callTool<StudioState>('app_get_studio_state', { project_name: proj });
    await applyState(state, { forceFormSync: true });
    setStatus(`Loaded project '${proj}'.`);
  });

  $<HTMLSelectElement>('scene-select').addEventListener('change', async (e) => {
    selectedSceneFilter = (e.target as HTMLSelectElement).value;
    await renderStoryboardGrid(currentState);
  });

  $<HTMLSelectElement>('shot-select').addEventListener('change', async (e) => {
    const proj = $<HTMLSelectElement>('project-select').value;
    const shot = (e.target as HTMLSelectElement).value;
    if (!proj || !shot) return;
    const state = await bridge.callTool<StudioState>('app_get_studio_state', {
      project_name: proj,
      shot_id: shot,
    });
    await applyState(state, { forceFormSync: true });
    setStatus(`Loaded shot '${proj}/${shot}'.`);
  });

  $<HTMLSelectElement>('branch-select').addEventListener('change', async (e) => {
    const idx = Number((e.target as HTMLSelectElement).value);
    const chosen = branchOptions[idx];
    if (!chosen) return;
    activeVideoPath = chosen.videoPath;
    if (chosen.ckpt) {
      activeCheckpointDir = chosen.ckpt.checkpoint_dir;
    }
    await loadTimelineVideo(activeVideoPath);
    setStatus(`Switched timeline preview to '${chosen.label}'.`);
  });

  $<HTMLButtonElement>('btn-refresh').addEventListener('click', async () => {
    const state = await bridge.callTool<StudioState>('app_get_studio_state', {
      project_name: $<HTMLSelectElement>('project-select').value || undefined,
      shot_id: $<HTMLSelectElement>('shot-select').value || undefined,
    });
    await applyState(state);
    setStatus('State & storyboard refreshed.');
  });

  $<HTMLSelectElement>('kf-character').addEventListener('change', (e) => {
    const val = (e.target as HTMLSelectElement).value;
    const repChar = $<HTMLSelectElement>('rep-character');
    if (val) repChar.value = val;
  });

  // Single-shot Keyframe Generation
  $<HTMLFormElement>('form-keyframes').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $<HTMLButtonElement>('btn-gen-keyframes');
    btn.disabled = true;
    try {
      const charName = $<HTMLSelectElement>('kf-character').value || 'ensemble';
      setStatus(`Generating First Frame for '${charName}' on RunPod...`);
      const useEnd = $<HTMLInputElement>('kf-use-end').checked;
      const state = await bridge.callTool<StudioState>('generate_shot_keyframes', {
        project_name: $<HTMLInputElement>('kf-project').value.trim(),
        shot_id: $<HTMLInputElement>('kf-shot').value.trim(),
        character_name: charName,
        start_prompt: $<HTMLTextAreaElement>('kf-start-prompt').value.trim(),
        end_prompt: useEnd ? $<HTMLTextAreaElement>('kf-end-prompt').value.trim() || undefined : undefined,
        lora_strength: Number($<HTMLInputElement>('kf-lora').value),
        seed: Number($<HTMLInputElement>('kf-seed').value),
      });
      await applyState(state, { forceFormSync: true });
      setStatus('First Frame generated. Validate or rebuild with editing suggestions before rendering video.');
    } catch (err) {
      setStatus(`Keyframe generation failed: ${(err as Error).message}`);
    } finally {
      btn.disabled = false;
    }
  });

  // Re-roll seed buttons
  for (const [btnId, which] of [['btn-reroll-start', 'start'], ['btn-reroll-end', 'end']] as const) {
    $<HTMLButtonElement>(btnId).addEventListener('click', async () => {
      try {
        setStatus(`Re-rolling ${which} keyframe seed...`);
        const promptOverride = which === 'start'
          ? $<HTMLTextAreaElement>('kf-start-prompt').value.trim() || undefined
          : $<HTMLTextAreaElement>('kf-end-prompt').value.trim() || undefined;
        const state = await bridge.callTool<StudioState>('regenerate_shot_keyframe', {
          project_name: $<HTMLInputElement>('kf-project').value.trim(),
          shot_id: $<HTMLInputElement>('kf-shot').value.trim(),
          which,
          prompt: promptOverride,
        });
        await applyState(state, { forceFormSync: true });
        setStatus(`Re-rolled ${which} keyframe.`);
      } catch (err) {
        setStatus(`Re-roll failed: ${(err as Error).message}`);
      }
    });
  }

  // Rebuild with Editing Suggestion (Kontext-anchored to current frame)
  for (const [btnId, inputId, which] of [
    ['btn-rebuild-start', 'edit-suggestion-start', 'start'],
    ['btn-rebuild-end', 'edit-suggestion-end', 'end'],
  ] as const) {
    $<HTMLButtonElement>(btnId).addEventListener('click', async () => {
      const suggestion = $<HTMLInputElement>(inputId).value.trim();
      if (!suggestion) {
        setStatus('Enter an editing suggestion first (or use Re-roll Seed).');
        return;
      }
      const btn = $<HTMLButtonElement>(btnId);
      btn.disabled = true;
      try {
        setStatus(`Rebuilding ${which} frame with suggestion via FLUX Kontext...`);
        const state = await bridge.callTool<StudioState>('regenerate_shot_keyframe', {
          project_name: $<HTMLInputElement>('kf-project').value.trim(),
          shot_id: $<HTMLInputElement>('kf-shot').value.trim(),
          which,
          edit_suggestion: suggestion,
        });
        await applyState(state, { forceFormSync: true });
        setStatus(`Rebuilt ${which} frame with editing suggestion.`);
      } catch (err) {
        setStatus(`Rebuild failed: ${(err as Error).message}`);
      } finally {
        btn.disabled = false;
      }
    });
  }

  // Area Selection & Rebuild modal (Brush Mask or SAM2 Text)
  for (const [btnId, which] of [['btn-mask-start', 'start'], ['btn-mask-end', 'end']] as const) {
    $<HTMLButtonElement>(btnId).addEventListener('click', async () => {
      const kfPath = which === 'start'
        ? currentState.shot_manifest?.keyframes.start_frame_path
        : currentState.shot_manifest?.keyframes.end_frame_path;
      if (!kfPath) {
        setStatus(`No ${which} keyframe image loaded yet.`);
        return;
      }
      activeMaskTarget = which;
      const dImg = $<HTMLImageElement>('dialog-kf-img');
      dImg.src = await bridge.resolveMediaUrl(kfPath);
      dImg.onload = () => {
        dialogPainter.resize(dImg.naturalWidth || 1280, dImg.naturalHeight || 720);
        dialogPainter.clear();
      };
      maskDialog.showModal();
    });
  }

  // Section 4: Approve First Frame & Render Video using dedicated Scene & Video Motion Direction prompt
  $<HTMLFormElement>('form-render').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $<HTMLButtonElement>('btn-approve-render');
    btn.disabled = true;
    try {
      const motionPrompt = $<HTMLTextAreaElement>('render-motion-prompt').value.trim();
      const audioCue = $<HTMLInputElement>('render-audio-cue').value.trim();
      const combinedMotion = audioCue ? `${motionPrompt} (${audioCue})` : motionPrompt;
      setStatus('First Frame approved! Unloading FLUX and rendering Wan 2.2 MoE video with Scene Direction...');
      const state = await bridge.callTool<StudioState>('approve_and_render_video', {
        project_name: $<HTMLInputElement>('kf-project').value.trim(),
        shot_id: $<HTMLInputElement>('kf-shot').value.trim(),
        prompt: combinedMotion || undefined,
        duration_sec: Number($<HTMLInputElement>('render-duration').value),
        driving_video_path: $<HTMLInputElement>('render-driving').value.trim() || undefined,
        audio_wav_path: $<HTMLInputElement>('render-wav').value.trim() || undefined,
        use_fast_distill_lora: $<HTMLInputElement>('render-fast-distill').checked,
      });
      await applyState(state, { forceFormSync: true });
      setStatus('Wan 2.2 shot rendered and ready for scrubbing in Timeline tab.');
    } catch (err) {
      setStatus(`Video render failed: ${(err as Error).message}`);
    } finally {
      btn.disabled = false;
    }
  });

  // Tab 2 scrubber & forms
  const slider = $<HTMLInputElement>('frame-slider');
  slider.addEventListener('input', () => void scrubToFrame(Number(slider.value)));
  $<HTMLButtonElement>('btn-step-m10').addEventListener('click', () => void scrubToFrame(Number(slider.value) - 10));
  $<HTMLButtonElement>('btn-step-m1').addEventListener('click', () => void scrubToFrame(Number(slider.value) - 1));
  $<HTMLButtonElement>('btn-step-p1').addEventListener('click', () => void scrubToFrame(Number(slider.value) + 1));
  $<HTMLButtonElement>('btn-step-p10').addEventListener('click', () => void scrubToFrame(Number(slider.value) + 10));

  $<HTMLButtonElement>('btn-toggle-play').addEventListener('click', () => {
    const v = $<HTMLVideoElement>('timeline-video');
    if (!v.hidden) {
      if (v.paused) void v.play();
      else v.pause();
    }
  });

  $<HTMLFormElement>('form-repair').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!activeVideoPath || !currentState.shot_manifest) {
      setStatus('Render or select a video first.');
      return;
    }
    const btn = $<HTMLButtonElement>('btn-repair-frame');
    btn.disabled = true;
    try {
      const frameIdx = Number($<HTMLInputElement>('frame-slider').value);
      const selectedChar =
        $<HTMLSelectElement>('rep-character').value || currentState.shot_manifest.character_name;
      let maskPath: string | undefined;
      if (timelinePainter.hasStrokes) {
        const saved = await bridge.callTool<{ mask_image_path: string }>('app_save_painted_mask', {
          project_name: currentState.shot_manifest.project_name,
          shot_id: currentState.shot_manifest.shot_id,
          mask_data_url: timelinePainter.exportBinaryMaskDataUrl(),
          label: `f${frameIdx}`,
        });
        maskPath = saved.mask_image_path;
      }
      setStatus(`Extracting & repairing Frame ${frameIdx} for '${selectedChar}'...`);
      const state = await bridge.callTool<StudioState>('extract_and_repair_frame', {
        source_video_path: currentState.shot_manifest.master_video_path || activeVideoPath,
        frame_index: frameIdx,
        character_name: selectedChar,
        repair_prompt: $<HTMLTextAreaElement>('rep-prompt').value.trim(),
        mask_target_text: $<HTMLInputElement>('rep-target-text').value.trim() || undefined,
        mask_image_path: maskPath,
        repair_engine: $<HTMLSelectElement>('rep-engine').value,
        inpaint_denoise: Number($<HTMLInputElement>('rep-denoise').value),
        project_name: currentState.shot_manifest.project_name,
        shot_id: currentState.shot_manifest.shot_id,
      });
      await applyState(state);
      setStatus(`Frame ${frameIdx} repaired. Inspect seam score and continue forward.`);
    } catch (err) {
      setStatus(`Frame repair failed: ${(err as Error).message}`);
    } finally {
      btn.disabled = false;
    }
  });

  $<HTMLFormElement>('form-continue').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!activeCheckpointDir || !currentState.shot_manifest) {
      setStatus('Repair a frame first to create a Frame K checkpoint.');
      return;
    }
    const btn = $<HTMLButtonElement>('btn-continue-branch');
    btn.disabled = true;
    try {
      const mode = $<HTMLSelectElement>('cont-mode').value;
      const selectedChar =
        $<HTMLSelectElement>('rep-character').value || currentState.shot_manifest.character_name;
      setStatus(`Rendering forward branch (mode=${mode})...`);
      const state = await bridge.callTool<StudioState>('continue_from_repaired_frame', {
        checkpoint_dir: activeCheckpointDir,
        character_name: selectedChar,
        mode,
        continuation_prompt: $<HTMLTextAreaElement>('cont-prompt').value.trim() || undefined,
        audio_wav_path: $<HTMLInputElement>('cont-wav').value.trim() || undefined,
        project_name: currentState.shot_manifest.project_name,
        shot_id: currentState.shot_manifest.shot_id,
      });
      await applyState(state);
      setStatus('Branch completed and stitched onto lossless prefix.');
    } catch (err) {
      setStatus(`Continuation failed: ${(err as Error).message}`);
    } finally {
      btn.disabled = false;
    }
  });

  try {
    const initial = await bridge.callTool<StudioState>('app_get_studio_state', {});
    await applyState(initial, { forceFormSync: true });
  } catch (err) {
    setStatus(`Initialized (${bridge.mode}). ${String((err as Error).message)}`);
  }
}

void init();
