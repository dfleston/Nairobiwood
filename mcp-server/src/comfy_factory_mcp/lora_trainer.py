"""Async detached LoRA training orchestrator (Qwen2.5-VL + ai-toolkit + musubi-tuner)."""

from __future__ import annotations

import json
import shlex
import uuid
from pathlib import Path

from comfy_factory_mcp.config import FactorySettings, resolve_safe_path
from comfy_factory_mcp.models import AsyncJobRecord, utc_now_iso
from comfy_factory_mcp.ssh_tunnel import SSHTunnelManager

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}


class LoRATrainingError(RuntimeError):
    """Raised when dataset validation or remote LoRA training fails."""


class LoRATrainer:
    """Manages ephemeral remote dataset upload, detached tmux training, polling, and Pull-and-Purge."""

    def __init__(self, settings: FactorySettings, tunnel: SSHTunnelManager | None = None) -> None:
        self.settings = settings
        self.tunnel = tunnel

    def _job_file(self, job_id: str) -> Path:
        return self.settings.jobs_dir / f"{job_id}.json"

    def save_job(self, record: AsyncJobRecord) -> Path:
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        record.updated_at = utc_now_iso()
        path = self._job_file(record.job_id)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_job(self, job_id: str) -> AsyncJobRecord:
        path = self._job_file(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown job_id: {job_id}")
        return AsyncJobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def validate_dataset(self, dataset_dir: Path) -> list[Path]:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise LoRATrainingError(f"Dataset directory not found: {dataset_dir}")
        images = [
            p for p in sorted(dataset_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not (15 <= len(images) <= 150):
            raise LoRATrainingError(
                f"Expected 15-150 reference images in {dataset_dir}, found {len(images)}."
            )
        return images

    @staticmethod
    def build_remote_training_script(
        *,
        job_id: str,
        character_name: str,
        trigger_word: str,
        remote_dataset_dir: str,
        remote_job_dir: str,
        train_flux: bool,
        train_wan22: bool,
        flux_steps: int = 2500,
        wan22_steps: int = 2500,
        flux_base_model: str = "unsloth/FLUX.1-dev",
        wan22_base_model: str = "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
    ) -> str:
        """Build the self-contained remote bash script executed inside detached tmux."""
        status_file = f"{remote_job_dir}/status.json"
        flux_cfg = f"{remote_job_dir}/flux_config.yaml"
        wan_cfg = f"{remote_job_dir}/wan22_config.yaml"
        flux_name = f"{character_name}_flux"
        wan_name = f"{character_name}_wan22"

        return f"""#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/nairobiwood/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" {shlex.quote(remote_job_dir)}

write_status() {{
  printf '{{"job_id":"%s","status":"%s","stage":"%s","progress_pct":%s,"error":%s}}\\n' \\
    {shlex.quote(job_id)} "$1" "$2" "$3" "$4" > {shlex.quote(status_file)}
}}
trap 'write_status "failed" "error" 0 "\\"Remote training command failed; inspect train.log\\""' ERR

# Patch PyTorch 2.6 custom_op list[torch.Tensor] schema check and TEXT_EMBED_CACHE_BATCH in ai-toolkit
python3 - <<'PATCH_PY'
from pathlib import Path
for p in Path('/opt/ai-toolkit').rglob('*.py'):
    try:
        txt = p.read_text(encoding='utf-8')
    except Exception:
        continue
    changed = False
    if 'torch.library.custom_op' in txt and 'list[torch.Tensor]' in txt:
        txt = txt.replace('list[torch.Tensor]', 'List[torch.Tensor]')
        lines = txt.splitlines(keepends=True)
        idx = 0
        while idx < len(lines) and (lines[idx].startswith('from __future__') or lines[idx].startswith('#') or not lines[idx].strip()):
            idx += 1
        lines.insert(idx, 'from typing import List\\n')
        txt = ''.join(lines)
        changed = True
    if p.name == 'dataloader_mixins.py' and 'TEXT_EMBED_CACHE_BATCH = ' in txt:
        import re
        txt = re.sub(r'TEXT_EMBED_CACHE_BATCH\\s*=\\s*\\d+', 'TEXT_EMBED_CACHE_BATCH = 2', txt)
        changed = True
    if p.name == 'wan22_14b_model.py':
        if 'if layer_offloading_transformer:' in txt and 'if not layer_offloading_transformer and self.model_config.low_vram:' not in txt:
            txt = txt.replace(
                '        return transformer',
                '        if not layer_offloading_transformer and self.model_config.low_vram:\\n'
                '            transformer_1.to("cpu")\\n'
                '            transformer_2.to("cpu")\\n'
                '            flush()\\n'
                '        return transformer',
                1,
            )
            changed = True
        old_to = (
            '    def to(self, *args, **kwargs) -> Self:\\n'
            '        # do not do to, this will be handled separately\\n'
            '        return self'
        )
        if old_to in txt:
            new_to = (
                '    def to(self, *args, **kwargs) -> Self:\\n'
                '        target = args[0] if args else kwargs.get("device", None)\\n'
                '        if target is not None and str(target) == "cpu":\\n'
                '            managed = (\\n'
                '                hasattr(self, "_memory_manager")\\n'
                '                or hasattr(self.transformer_1, "_memory_manager")\\n'
                '                or hasattr(self.transformer_2, "_memory_manager")\\n'
                '            )\\n'
                '            if not managed:\\n'
                '                self.transformer_1.to("cpu")\\n'
                '                self.transformer_2.to("cpu")\\n'
                '                torch.cuda.empty_cache()\\n'
                '        return self'
            )
            txt = txt.replace(old_to, new_to, 1)
            changed = True
    if changed:
        p.write_text(txt, encoding='utf-8')
PATCH_PY

write_status "running" "captioning_dataset" 10 "null"
python3 - <<'PY'
import glob, os
dataset_dir = {remote_dataset_dir!r}
trigger = {trigger_word!r}
for img in sorted(glob.glob(os.path.join(dataset_dir, "*.*")) + glob.glob(os.path.join(dataset_dir, "videos", "*.*"))):
    if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm")):
        txt = os.path.splitext(img)[0] + ".txt"
        if not os.path.exists(txt):
            with open(txt, "w", encoding="utf-8") as fh:
                fh.write(f"{{trigger}}, detailed facial portrait, natural skin texture, consistent identity\\n")
PY

if [ "{1 if train_flux else 0}" = "1" ]; then
  write_status "running" "training_flux_ai_toolkit" 25 "null"
  cat > {shlex.quote(flux_cfg)} <<'YAML_FLUX'
job: extension
config:
  name: "{flux_name}"
  process:
    - type: 'sd_trainer'
      training_folder: "{remote_job_dir}/out"
      performance_log_every: 50
      device: cuda:0
      trigger_word: "{trigger_word}"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 10
        push_to_hub: false
      datasets:
        - folder_path: "{remote_dataset_dir}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ 512, 768, 1024 ]
      train:
        batch_size: 1
        steps: {flux_steps}
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        skip_first_sample: true
        disable_sampling: false
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "{flux_base_model}"
        is_flux: true
        quantize: true
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
          - "{trigger_word}, close-up cinematic portrait looking directly at the camera, bare brown eyes, straight-back braided cornrow hair on top, closely faded gray temples, short salt-and-pepper beard with a gray soul patch, wearing a dark navy henley shirt, natural window lighting, 85mm lens"
          - "{trigger_word} driving a car in Nairobi, frontal view looking through the windshield at the driver, bare brown eyes focused on the road, hands on the steering wheel, straight-back braided cornrow hair, faded gray temples, short salt-and-pepper beard, wearing a dark charcoal jacket, natural daylight"
          - "{trigger_word}, three-quarter profile portrait outdoors at golden hour, bare brown eyes, straight-back braided cornrow hair on top, closely faded gray temples, short salt-and-pepper beard with a gray soul patch, wearing a black crew-neck t-shirt, shallow depth of field"
        neg: ""
        seed: 42
        walk_seed: false
        guidance_scale: 3.5
        sample_steps: 24
meta:
  name: "[name]"
  version: '1.0'
YAML_FLUX
  PYTHONPATH=/opt/ai-toolkit python3 /opt/ai-toolkit/run.py {shlex.quote(flux_cfg)}
  cp -f "{remote_job_dir}/out/{flux_name}/{flux_name}.safetensors" "{remote_job_dir}/{flux_name}.safetensors"
  mkdir -p "{remote_job_dir}/checkpoints" "{remote_job_dir}/samples"
  cp -f "{remote_job_dir}/out/{flux_name}"/*000*.safetensors "{remote_job_dir}/checkpoints/" 2>/dev/null || true
  cp -f "{remote_job_dir}/out/{flux_name}/samples"/* "{remote_job_dir}/samples/" 2>/dev/null || true
fi

if [ "{1 if train_wan22 else 0}" = "1" ]; then
  write_status "running" "training_wan22_i2v_low_noise" 65 "null"
  VIDEO_DATASET_BLOCK=""
  if compgen -G "{remote_dataset_dir}/videos/*.mp4" > /dev/null; then
    VIDEO_DATASET_BLOCK="$(cat <<'YAML_VID'
        - folder_path: "{remote_dataset_dir}/videos"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          num_frames: 17
          fps: 16
          num_repeats: 3
          resolution: [ 512 ]
YAML_VID
)"
  fi
  cat > {shlex.quote(wan_cfg)} <<YAML_WAN
job: extension
config:
  name: "{wan_name}"
  process:
    - type: 'sd_trainer'
      training_folder: "{remote_job_dir}/out"
      performance_log_every: 50
      device: cuda:0
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 10
        push_to_hub: false
      datasets:
        - folder_path: "{remote_dataset_dir}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          num_frames: 1
          resolution: [ 512, 768, 1024 ]
$VIDEO_DATASET_BLOCK
      train:
        batch_size: 1
        steps: {wan22_steps}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        timestep_type: 'linear'
        optimizer: "adamw8bit"
        lr: 1e-4
        optimizer_params:
          weight_decay: 1e-4
        skip_first_sample: true
        disable_sampling: true
        dtype: bf16
        switch_boundary_every: 10
        cache_text_embeddings: true
      model:
        name_or_path: "{wan22_base_model}"
        arch: 'wan22_14b_i2v'
        quantize: true
        qtype: "float8"
        quantize_te: true
        qtype_te: "float8"
        low_vram: true
        model_kwargs:
          train_high_noise: false
          train_low_noise: true
meta:
  name: "[name]"
  version: '1.0'
YAML_WAN
  PYTHONPATH=/opt/ai-toolkit python3 /opt/ai-toolkit/run.py {shlex.quote(wan_cfg)}
  for f in "{remote_job_dir}/out/{wan_name}"/*.safetensors; do
    [ -e "$f" ] || continue
    case "$f" in
      *0000*.safetensors) continue ;;
      *) cp -f "$f" "{remote_job_dir}/" ;;
    esac
  done
fi

write_status "completed" "ready_to_collect" 100 "null"
"""

    async def start_lora_training(
        self,
        *,
        character_name: str,
        local_dataset_dir: str,
        trigger_word: str,
        train_flux: bool = True,
        train_wan22: bool = True,
    ) -> AsyncJobRecord:
        if self.tunnel is None:
            raise LoRATrainingError("Cloud GPU factory is not active. Run factory_up first.")

        dataset_path = resolve_safe_path(local_dataset_dir, self.settings, must_exist=True)
        self.validate_dataset(dataset_path)

        job_id = f"lora_{character_name}_{uuid.uuid4().hex[:8]}"
        remote_dataset_dir = f"/dev/shm/factory_scratch/datasets/{job_id}"
        remote_job_dir = f"/dev/shm/factory_scratch/jobs/{job_id}"
        tmux_session = f"nw_{job_id[:20]}"

        await self.tunnel.rsync_upload(dataset_path, remote_dataset_dir)
        script_body = self.build_remote_training_script(
            job_id=job_id,
            character_name=character_name,
            trigger_word=trigger_word,
            remote_dataset_dir=remote_dataset_dir,
            remote_job_dir=remote_job_dir,
            train_flux=train_flux,
            train_wan22=train_wan22,
        )
        remote_script = f"{remote_job_dir}/run_train.sh"
        remote_log = f"{remote_job_dir}/train.log"
        tmux_cmd = f"bash {shlex.quote(remote_script)} > {shlex.quote(remote_log)} 2>&1"
        await self.tunnel.run_remote(
            f"mkdir -p {shlex.quote(remote_job_dir)} && chmod 700 {shlex.quote(remote_job_dir)} && "
            f"cat > {shlex.quote(remote_script)} <<'NW_EOF'\n{script_body}\nNW_EOF\n"
            f"chmod +x {shlex.quote(remote_script)} && "
            f"tmux new-session -d -s {shlex.quote(tmux_session)} {shlex.quote(tmux_cmd)}"
        )

        record = AsyncJobRecord(
            job_id=job_id,
            character_name=character_name,
            trigger_word=trigger_word,
            train_flux=train_flux,
            train_wan22=train_wan22,
            remote_job_dir=remote_job_dir,
            remote_dataset_dir=remote_dataset_dir,
            tmux_session=tmux_session,
            status="running",
            stage="launched",
            progress_pct=5.0,
        )
        self.save_job(record)
        return record

    async def get_job_status(self, job_id: str) -> AsyncJobRecord:
        record = self.load_job(job_id)
        if record.status in ("completed", "collected") or self.tunnel is None:
            return record

        status_path = f"{record.remote_job_dir}/status.json"
        raw = await self.tunnel.run_remote(
            f"if [ -f {shlex.quote(status_path)} ]; then cat {shlex.quote(status_path)}; fi"
        )
        if raw.strip():
            payload = json.loads(raw.strip())
            record.status = payload.get("status", record.status)
            record.stage = payload.get("stage", record.stage)
            record.progress_pct = float(payload.get("progress_pct", record.progress_pct))
            record.error = payload.get("error")
            self.save_job(record)

        # Opportunistically sync any checkpoint sample images and intermediate .safetensors generated so far
        try:
            local_lora_dir = self.settings.loras_dir / record.character_name
            local_samples_dir = local_lora_dir / "samples"
            local_ckpts_dir = local_lora_dir / "checkpoints"
            local_samples_dir.mkdir(parents=True, exist_ok=True)
            local_ckpts_dir.mkdir(parents=True, exist_ok=True)
            remote_samples = f"{record.remote_job_dir}/out/{record.character_name}_flux/samples"
            has_samples = await self.tunnel.run_remote(
                f"if [ -d {shlex.quote(remote_samples)} ] && [ -n \"$(ls -A {shlex.quote(remote_samples)} 2>/dev/null)\" ]; then echo yes; fi"
            )
            if has_samples.strip() == "yes":
                await self.tunnel.rsync_download(f"{remote_samples}/*", local_samples_dir)

            for stage_sub in (f"{record.character_name}_flux", f"{record.character_name}_wan22"):
                remote_out = f"{record.remote_job_dir}/out/{stage_sub}"
                has_ckpts = await self.tunnel.run_remote(
                    f"if compgen -G {shlex.quote(remote_out)}/*.safetensors > /dev/null; then echo yes; fi"
                )
                if has_ckpts.strip() == "yes":
                    await self.tunnel.rsync_download(f"{remote_out}/*.safetensors", local_ckpts_dir)
        except Exception:
            pass
        return record

    async def collect_job_artifacts(self, job_id: str) -> AsyncJobRecord:
        """Download trained .safetensors weights to ~/ComfyFactory/loras/<character>/ and shred remote scratch."""
        if self.tunnel is None:
            raise LoRATrainingError("Cloud GPU factory is not active.")
        record = await self.get_job_status(job_id)
        if record.status not in ("completed", "collected"):
            raise LoRATrainingError(f"Job {job_id} is '{record.status}', not ready to collect.")

        local_lora_dir = self.settings.loras_dir / record.character_name
        local_lora_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(local_lora_dir, 0o700)

        await self.tunnel.rsync_download(
            f"{record.remote_job_dir}/{record.character_name}*.safetensors",
            local_lora_dir,
        )
        try:
            local_ckpts_dir = local_lora_dir / "checkpoints"
            local_ckpts_dir.mkdir(parents=True, exist_ok=True)
            await self.tunnel.rsync_download(
                f"{record.remote_job_dir}/checkpoints/",
                local_ckpts_dir,
            )
        except Exception:
            pass
        await self.tunnel.purge_remote(record.remote_dataset_dir)
        await self.tunnel.purge_remote(record.remote_job_dir)

        artifacts = [str(p) for p in sorted(local_lora_dir.glob("*.safetensors"))]
        record.local_artifacts = artifacts
        record.status = "collected"
        record.stage = "purged_remote"
        record.progress_pct = 100.0
        self.save_job(record)
        return record
