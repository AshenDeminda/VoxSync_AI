import os
from pathlib import Path
from f5_tts.api import F5TTS
import src.config as config

class TTSEngine:
    def __init__(self):
        # F5TTS will automatically respect the HF_HOME env variable we set in config.py
        # It will load 'models--SWivid--F5-TTS' and 'models--charactr--vocos-mel-24khz' from D:\
        self.model = F5TTS()
        self.output_dir = Path("data/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, text: str, ref_audio: str) -> str:
        output_file = self.output_dir / "response.wav"
        
        self.model.infer(
            ref_file=ref_audio,
            ref_text="", # Auto-detect reference text
            gen_text=text,
            file_wave=str(output_file),
            remove_silence=True
        )
        return str(output_file)