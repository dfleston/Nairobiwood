#!/usr/bin/env bash
set -euo pipefail

JOB_DIR=/dev/shm/factory_scratch/jobs/lora_D_Leston_wan22
DATASET_DIR=/dev/shm/factory_scratch/datasets/lora_D_Leston_wan22
mkdir -p "$JOB_DIR/out"

cat > "$JOB_DIR/wan22_config.yaml" <<'YAML'
job: extension
config:
  name: "D_Leston_wan22"
  process:
    - type: 'sd_trainer'
      training_folder: "/dev/shm/factory_scratch/jobs/lora_D_Leston_wan22/out"
      performance_log_every: 25
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
        - folder_path: "/dev/shm/factory_scratch/datasets/lora_D_Leston_wan22"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          num_frames: 1
          resolution: [ 512, 768, 1024 ]
        - folder_path: "/dev/shm/factory_scratch/datasets/lora_D_Leston_wan22/videos"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          num_frames: 17
          fps: 16
          num_repeats: 3
          resolution: [ 512 ]
      train:
        batch_size: 1
        steps: 1500
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
        name_or_path: "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16"
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
YAML

cat > "$JOB_DIR/run_train.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/nairobiwood/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/nairobiwood/ai-toolkit
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

JOB_DIR=/dev/shm/factory_scratch/jobs/lora_D_Leston_wan22
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Wan 2.2 LoRA Training (resuming to 1500 steps, eager mode)..."

python3 /workspace/nairobiwood/ai-toolkit/run.py "$JOB_DIR/wan22_config.yaml"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished. Copying final weights..."
for f in "$JOB_DIR/out/D_Leston_wan22"/*.safetensors; do
  [ -e "$f" ] || continue
  case "$f" in
    *0000*.safetensors) continue ;;
    *) cp -f "$f" "$JOB_DIR/" ;;
  esac
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
SH

chmod +x "$JOB_DIR/run_train.sh"
echo "Job configuration created successfully at $JOB_DIR"
