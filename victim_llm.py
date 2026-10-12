#!/usr/bin/env python3
"""Target LLM wrapper for Speech/Audio-Omni Models:
- Qwen/Qwen2.5-Omni-7B
- openbmb/MiniCPM-o-2_6
- THUDM/glm-4-voice-9b
- ICTNLP/Llama-3.1-8B-Omni
"""

import os
from pathlib import Path
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCausalLM
from config import CFG

class VictimLLM:
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B"):
        self.model_name = model_name
        self.device = CFG.device
        print(f"Loading victim audio LLM: {model_name}")

        self.output_dir = Path("outputs/response_audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.response_counter = 1

        # Load Processor and Model with trust_remote_code=True for custom Omni architectures
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if CFG.use_bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response(self, audio_path, max_new_tokens=150):
        """
        Accepts an input .wav file path (e.g., from tts.generate_wav()),
        runs speech-to-speech inference, and saves the output audio to 
        outputs/response_audio/response_X.wav.
        """
        # 1. Load input waveform
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure single channel (mono) if required by processor
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Process audio input through processor
        inputs = self.processor(
            audios=waveform.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        # 3. Generate response audio tokens
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if CFG.use_bf16 else torch.float16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

        # 4. Extract generated audio array/waveform from output
        # (Depending on processor type, decode or extract raw speech tensor)
        if hasattr(self.processor, "batch_decode_speeches"):
            audio_out = self.processor.batch_decode_speeches(outputs)[0]
        elif hasattr(outputs, "audio"):
            audio_out = outputs.audio[0]
        else:
            # Fallback for models returning raw waveform tensor directly
            audio_out = outputs[0]

        # Ensure tensor format (1, num_samples)
        if not isinstance(audio_out, torch.Tensor):
            audio_out = torch.tensor(audio_out)
        if audio_out.ndim == 1:
            audio_out = audio_out.unsqueeze(0)

        # 5. Define output filepath (response_1.wav, response_2.wav, ...)
        save_path = self.output_dir / f"response_{self.response_counter}.wav"
        self.response_counter += 1

        # Target sample rate (defaulting to 24kHz common for speech generation)
        out_sample_rate = getattr(self.model.config, "audio_sample_rate", 24000)
        
        # Save output audio file
        torchaudio.save(str(save_path), audio_out.cpu().float(), out_sample_rate)
        
        print(f"Saved response audio to: {save_path}")
        return str(save_path)
