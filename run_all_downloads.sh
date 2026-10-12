#!/bin/bash
set -euo pipefail

cd /home/udaykc_iitp/vivek/ARed
export HF_HUB_DISABLE_XET=1
PYTHON_BIN="/home/udaykc_iitp/.conda/envs/spectre/bin/python3"
while pgrep -f "mistralai/Mistral-7B-Instruct-v0.3" > /dev/null; do
    echo "[$(date)] Waiting for Mistral-7B download to complete..."
    sleep 10
done
echo "[$(date)] Mistral-7B finished. Now downloading remaining models..."
$PYTHON_BIN download_remaining.py
echo "[$(date)] All models successfully downloaded."
