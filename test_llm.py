
import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class VictimLLM:
    def __init__(self, model_id="Qwen/Qwen2.5-Omni-3B"):
        print("Loading victim audio LLM:", model_id)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    def process_audio(self, input_wav_path, output_wav_path="output_response.wav", speaker="Chelsie"):
        # Set mandatory system prompt required for audio response generation
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": input_wav_path}
                ],
            },
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        ).to(self.model.device)

        # Generate output (text_ids and audio waveform tensor)
        text_ids, audio_tensor = self.model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=True,
            speaker=speaker
        )

        # Save audio output directly to .wav file (sampling rate is 24000 Hz)
        audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()
        sf.write(output_wav_path, audio_np, samplerate=24000)

        print(f"Audio response saved to: {output_wav_path}")
        return output_wav_path
