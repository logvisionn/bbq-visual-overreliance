#!/usr/bin/env bash
# =============================================================================
# setup_runpod.sh  —  one-time environment bootstrap for the RunPod pod
# Run once after SSH-ing in:  bash setup_runpod.sh
# =============================================================================
set -euo pipefail

echo "=================================================="
echo "  BBQ Visual Over-reliance — RunPod Setup"
echo "=================================================="

# ── 1. System packages ────────────────────────────────
echo "[1/4] System packages..."
apt-get update -qq
apt-get install -y -qq zip unzip tmux git htop

# ── 2. Python dependencies (exact versions from notebooks) ────────────────────
echo "[2/4] Python packages..."
pip install -q --upgrade pip

pip install -q \
  "torch==2.3.1" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install -q \
  "transformers==4.57.1" "tokenizers==0.22.1" "huggingface_hub==0.36.0" \
  "diffusers==0.30.0" "accelerate>=0.30.0" "bitsandbytes>=0.43.1" \
  sentencepiece safetensors \
  "numpy==1.26.4" "pandas==2.2.2" "pillow==11.1.0" \
  "datasets==2.19.0" "pyarrow==14.0.2" "statsmodels>=0.14.2" tqdm

echo "Packages installed."

# ── 3. Pre-download all four models ───────────────────
echo "[3/4] Pre-downloading models (10-20 min)..."
python3 - <<'EOF'
import os, torch
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("  [1/4] stabilityai/sd-turbo + safety checker (Experiment 1)...")
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16)
print("  sd-turbo OK.")

print("  [2/4] stabilityai/sdxl-turbo (Experiment 2)...")
from diffusers import AutoPipelineForText2Image
AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
print("  sdxl-turbo OK.")

print("  [3/4] Qwen/Qwen2-VL-2B-Instruct (Experiment 1)...")
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16,
    quantization_config=bnb, trust_remote_code=True)
print("  Qwen2-VL-2B OK.")

print("  [4/4] Qwen/Qwen2-VL-7B-Instruct (Experiment 2)...")
AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16,
    quantization_config=bnb, trust_remote_code=True)
print("  Qwen2-VL-7B OK.")
print("All models cached successfully.")
EOF

# ── 4. GPU check ──────────────────────────────────────
echo "[4/4] GPU check..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  GPU : {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

echo ""
echo "=================================================="
echo "  Setup complete! Run experiments with:"
echo "  tmux new -s bbq"
echo "  python run_experiment1.py"
echo "  python run_experiment2.py"
echo "=================================================="
