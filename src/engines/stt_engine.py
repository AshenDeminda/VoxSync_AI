from faster_whisper import WhisperModel
import src.config as config  # Imports our cache settings

class STTEngine:
    def __init__(self):
        # Because we set HF_HOME in config.py, this will grab from your D: drive
        self.model = WhisperModel(
            config.WHISPER_MODEL, 
            device=config.WHISPER_DEVICE, 
            compute_type="float16" # Use "int8" if you have low VRAM
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()