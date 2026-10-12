from pathlib import Path
import os
import re

import torch
from faster_whisper import WhisperModel


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "transcripts"

MODEL_ID = "Systran/faster-whisper-small.en"

print(f"Base dir : {BASE_DIR}")
print(f"Output dir : {OUTPUT_DIR}")


class LocalSTT:
    def __init__(self, model_id=MODEL_ID, device=None, compute_type=None):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"[STT] Loading model: {model_id}")
        print(f"[STT] Device: {device}")
        print(f"[STT] Compute type: {compute_type}")

        self.model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
        )

        print("[STT] Model loaded.")

    @staticmethod
    def _next_output_path():
        """Find the next available transcript_N.txt filename."""

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        pattern = re.compile(r"^transcript_(\d+)\.txt$")
        existing_numbers = []

        for path in OUTPUT_DIR.iterdir():
            match = pattern.match(path.name)
            if match:
                existing_numbers.append(int(match.group(1)))

        next_number = max(existing_numbers, default=0) + 1
        output_path = OUTPUT_DIR / f"transcript_{next_number}.txt"

        while output_path.exists():
            next_number += 1
            output_path = OUTPUT_DIR / f"transcript_{next_number}.txt"

        return output_path

    def generate_txt(
        self,
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        output_path=None,
    ):
        """
        Convert victim response audio to text.

        Saves each transcript to a unique text file.

        Returns:
            str: transcript to pass to the AUTORED pipeline.
        """

        audio_path = str(audio_path)

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Victim response audio not found: {audio_path}"
            )

        segments, _ = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        if output_path is None:
            output_path = self._next_output_path()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(transcript + "\n")

        print(f"[STT] Transcript saved: {output_path}")

        return transcript


# Load the STT model only once per Python process.
_stt = None


def generate_txt(
    audio_path,
    language="en",
    device=None,
    compute_type=None,
    output_path=None,
):
    """
    Main function for run_autored.py.

    Returns the transcript string and saves it to a unique file.
    """

    global _stt

    if _stt is None:
        _stt = LocalSTT(
            device=device,
            compute_type=compute_type,
        )

    return _stt.generate_txt(
        audio_path=audio_path,
        language=language,
        output_path=output_path,
    )
