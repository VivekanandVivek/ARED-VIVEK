#!/bin/bash
#SBATCH --job-name=autored_vivek
#SBATCH --partition=gpu
#SBATCH --account=iitp
#SBATCH --qos=iitp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_autored_%j.out
#SBATCH --error=slurm_autored_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=========================================================="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID:       ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "CPUs allocated:     ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "=========================================================="

# ------------------------------------------------------------
# 1. CUDA & Environment Setup
# ------------------------------------------------------------
module load cuda/12.8 2>/dev/null || true
if command -v nvcc &>/dev/null; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    echo "CUDA_HOME set to: ${CUDA_HOME}"
fi

# ------------------------------------------------------------
# 2. HuggingFace & PyTorch Configuration
# ------------------------------------------------------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token)}"
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ------------------------------------------------------------
# 3. Conda Environment Activation
# ------------------------------------------------------------
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate /home/udaykc_iitp/.conda/envs/spectre || conda activate spectre

# ------------------------------------------------------------
# 4. Working Directory & Output Folders
# ------------------------------------------------------------
cd "/home/udaykc_iitp/vivek/ARed"
mkdir -p outputs/{sft,detector,extractor,rl,figures,evaluation,full_evaluation}

# ------------------------------------------------------------
# 5. GPU Detection & Visibility
# ------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    echo "--- Available GPU Devices ---"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

    # If CUDA_VISIBLE_DEVICES is unset or exposes multiple devices, pick the one with most free VRAM
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* ]]; then
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -rn | head -n 1 | cut -d, -f1 | tr -d ' ')
        if [ -n "$BEST_GPU" ]; then
            export CUDA_VISIBLE_DEVICES=$BEST_GPU
            echo "Selected GPU index with most free VRAM: GPU $CUDA_VISIBLE_DEVICES"
        fi
    fi
fi
echo "Active CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-None}"
echo "Python binary: $(which python3)"
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo "=========================================================="

# ------------------------------------------------------------
# 6. Pipeline Execution
# ------------------------------------------------------------

# Step 1: Supervised Fine-Tuning (Attacker)
echo "--- [1/4] Running SFT Training ---"
python3 -u train_sft.py

# Step 2: Stop-Point Identifier (BERT Detector)
echo "--- [2/4] Running Detector Training ---"
python3 -u train_detector.py

# Step 3: Sensitive Info Extractor (T5)
echo "--- [3/4] Running Extractor Training ---"
python3 -u train_extractor.py

# Step 4: NLPO/PPO RL Training & 4-LLM CTF Evaluation
# Pass input '2' to non-interactively run full four-LLM evaluation
echo "--- [4/4] Running RL Training & Evaluation ---"
echo "2" | python3 -u run_autored.py

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
