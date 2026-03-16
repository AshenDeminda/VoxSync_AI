import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os

# Target sample rates
TARGET_SR = 16000  # For Whisper (STT)
TTS_SR = 24000     # For F5-TTS (Voice Cloning)

def process_audio(audio_path: str, target_sr: int) -> Path:
    """Loads, resamples, trims silence, and saves the audio."""
    path = Path(audio_path)
    
    # 1. Load and resample
    waveform, sr = librosa.load(str(path), sr=target_sr, mono=True, dtype=np.float32)
    
    # 2. Trim silence
    trimmed, _ = librosa.effects.trim(waveform, top_db=30.0)
    
    # 3. Peak normalize (make it loud and clear)
    peak = np.max(np.abs(trimmed))
    if peak > 0:
        trimmed = trimmed / peak * 0.99
        
    # 4. Save processed file
    output_path = path.with_name(path.stem + "_processed.wav")
    sf.write(str(output_path), trimmed, target_sr, subtype="PCM_16")
    
    return output_path

def preprocess_for_stt(audio_path: str) -> str:
    return str(process_audio(audio_path, TARGET_SR))

def preprocess_for_tts(audio_path: str) -> str:
    return str(process_audio(audio_path, TTS_SR))