#!/usr/bin/env python3
import time
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, snapshot_download

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
api = HfApi()

print(f"[{time.strftime('%X')}] Monitoring approval status for {model_id}...", flush=True)
while True:
    try:
        api.hf_hub_download(model_id, "config.json")
        print(f"\n[{time.strftime('%X')}] ACCESS APPROVED! Starting download of {model_id}...", flush=True)
        break
    except Exception as e:
        err_msg = str(e).lower()
        if "awaiting a review" in err_msg or "restricted" in err_msg or "403" in err_msg:
            print(f"[{time.strftime('%X')}] Still awaiting review from Meta. Retrying in 30s...", flush=True)
            time.sleep(30)
        else:
            print(f"[{time.strftime('%X')}] Transient error: {e}. Retrying in 30s...", flush=True)
            time.sleep(30)

snapshot_download(
    model_id,
    allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "special_tokens*"],
    resume_download=True
)
print(f"[{time.strftime('%X')}] SUCCESS: {model_id} fully downloaded and cached!", flush=True)
