
#!/usr/bin/env python3
"""AUTORED Local TTS: Convert full_prompt directly to a WAV file."""

from pathlib import Path
import re
import wave
import os
from huggingface_hub import hf_hub_download
from piper import PiperVoice


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "piper"
OUTPUT_DIR = BASE_DIR / "audio"

HF_REPO = "rhasspy/piper-voices"
MODEL_FILE = "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
CONFIG_FILE = "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"


class LocalTTS:
    def __init__(self, use_cuda=False):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.model_path = MODEL_DIR / Path(MODEL_FILE).name
        self.config_path = MODEL_DIR / Path(CONFIG_FILE).name

        self._download_if_needed()

        print(f"[TTS] Loading model: {self.model_path}")
        self.voice = PiperVoice.load(
            str(self.model_path),
            config_path=str(self.config_path),
            use_cuda=use_cuda,
        )
        print("[TTS] Model loaded.")

    def _download_if_needed(self):
        for filename, local_path in (
            (MODEL_FILE, self.model_path),
            (CONFIG_FILE, self.config_path),
        ):
            if not local_path.exists():
                print(f"[TTS] Downloading {filename}...")
                downloaded = Path(
                    hf_hub_download(
                        repo_id=HF_REPO,
                        filename=filename,
                        local_dir=str(MODEL_DIR),
                    )
                )

                if downloaded != local_path:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    downloaded.replace(local_path)

    @staticmethod
    def _next_output_path():
        """Return the next available input_N.wav filename."""
        pattern = re.compile(r"^input_(\d+)\.wav$")
        numbers = []

        for path in OUTPUT_DIR.iterdir():
            match = pattern.match(path.name)
            if match:
                numbers.append(int(match.group(1)))

        number = max(numbers, default=0) + 1
        output_path = OUTPUT_DIR / f"input_{number}.wav"

        while output_path.exists():
            number += 1
            output_path = OUTPUT_DIR / f"input_{number}.wav"

        return output_path

    def generate_wav(self, full_prompt, output_path=None):
        """Convert the supplied full_prompt directly into a WAV file."""
        if not isinstance(full_prompt, str):
            raise TypeError("full_prompt must be a string.")

        if not full_prompt.strip():
            raise ValueError("full_prompt cannot be empty.")

        if output_path is None:
            output_path = self._next_output_path()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), "wb") as wav_file:
            self.voice.synthesize_wav(full_prompt, wav_file)

        print(f"[TTS] Audio saved: {output_path}")
        return str(output_path)


# Load the TTS model only once per Python process.
_tts = None


def generate_wav(full_prompt, output_path=None, use_cuda=False):
    """Generate audio directly from full_prompt."""
    global _tts

    if _tts is None:
        _tts = LocalTTS(use_cuda=use_cuda)

    return _tts.generate_wav(full_prompt=full_prompt,output_path=output_path)
