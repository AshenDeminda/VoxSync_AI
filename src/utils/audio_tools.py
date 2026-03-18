import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os
import shutil
import subprocess


def _find_ffmpeg_exe() -> str | None:
    """Return an ffmpeg executable path if available.

    Prefers a system-installed ffmpeg on PATH, but also supports the
    `imageio-ffmpeg` bundled binary when installed.
    """
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        return ffmpeg_on_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _decode_with_ffmpeg(input_path: Path, *, target_sr: int) -> Path:
    ffmpeg_exe = _find_ffmpeg_exe()
    if not ffmpeg_exe:
        raise ValueError(
            "FFmpeg is required to decode this audio (browser mic recordings are usually WebM/Opus; reference audio may be MP3). "
            "Install FFmpeg or run `pip install imageio-ffmpeg` and restart the server."
        )

    output_path = input_path.with_name(f"{input_path.stem}_decoded_{target_sr}.wav")

    # Convert to mono PCM WAV at the desired sample rate.
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        str(output_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        if len(tail) > 2000:
            tail = tail[-2000:]
        raise ValueError(
            f"FFmpeg failed to decode '{input_path.name}'. "
            f"{tail if tail else 'Please verify the uploaded audio file is valid.'}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError(f"Decoded audio was empty for '{input_path.name}'.")

    return output_path

# Target sample rates
TARGET_SR = 16000  # For Whisper (STT)
TTS_SR = 24000     # For F5-TTS (Voice Cloning)

def process_audio(audio_path: str, target_sr: int) -> Path:
    """Loads, resamples, trims silence, and saves the audio."""
    path = Path(audio_path)

    if not path.exists():
        raise ValueError(f"Audio file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {path.name}")
    
    # 1. Load and resample
    # Many browser mic recordings are WebM/Opus but get named .wav; libsndfile
    # can't read them. Avoid librosa's audioread fallback (deprecated + noisy)
    # by proactively decoding unsupported formats with ffmpeg.
    load_path = path
    try:
        sf.info(str(path))
    except Exception:
        load_path = _decode_with_ffmpeg(path, target_sr=target_sr)

    try:
        waveform, sr = librosa.load(str(load_path), sr=target_sr, mono=True, dtype=np.float32)
    except Exception:
        # As a last resort, attempt an ffmpeg decode even if sf.info() passed.
        decoded = _decode_with_ffmpeg(path, target_sr=target_sr)
        waveform, sr = librosa.load(str(decoded), sr=target_sr, mono=True, dtype=np.float32)

    if waveform.size == 0:
        raise ValueError(f"No audio samples found in '{path.name}'.")
    
    # 2. Trim silence
    trimmed, _ = librosa.effects.trim(waveform, top_db=30.0)

    if trimmed.size == 0:
        raise ValueError(f"Audio appears to be all silence: {path.name}")
    
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