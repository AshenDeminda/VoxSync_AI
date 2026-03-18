import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force HuggingFace to use your D: drive cache
os.environ["HF_HOME"] = os.getenv("HF_HOME", r"D:\VoxSync_Cache\huggingface")

# Ensure an ffmpeg command is discoverable for any library that relies on it
# (e.g., pydub/librosa/audioread). On Windows, `imageio-ffmpeg` ships a
# differently named binary (not `ffmpeg.exe`), so we generate a tiny wrapper
# script and add it to PATH for this process.
try:
    import imageio_ffmpeg  # type: ignore

    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if os.name == "nt":
        _repo_root = Path(__file__).resolve().parent.parent
        _wrapper_dir = _repo_root / "data" / "temp" / "_bin"
        _wrapper_dir.mkdir(parents=True, exist_ok=True)

        # pydub only searches for ffmpeg.exe on Windows (not .bat/.cmd), so we
        # create a local alias copy named exactly `ffmpeg.exe`.
        _ffmpeg_alias = _wrapper_dir / "ffmpeg.exe"
        try:
            src_size = Path(_ffmpeg_exe).stat().st_size
            dst_size = _ffmpeg_alias.stat().st_size if _ffmpeg_alias.exists() else -1
            if dst_size != src_size:
                shutil.copyfile(_ffmpeg_exe, _ffmpeg_alias)
        except Exception:
            # If we can't create the alias, we'll still fall back to explicit
            # ffmpeg invocation in audio_tools.
            pass

        _wrapper_dir_str = str(_wrapper_dir)
        if _wrapper_dir_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _wrapper_dir_str + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
_raw_ollama_host = (os.getenv("OLLAMA_HOST") or "").strip()
if not _raw_ollama_host:
    OLLAMA_HOST = "http://127.0.0.1:11434"
else:
    # 0.0.0.0 is a listen/bind address, not a valid destination for clients.
    # If a user has it set globally, map it to loopback so the client can connect.
    if _raw_ollama_host.startswith("http://0.0.0.0"):
        OLLAMA_HOST = "http://127.0.0.1" + _raw_ollama_host[len("http://0.0.0.0"):]
    elif _raw_ollama_host.startswith("https://0.0.0.0"):
        OLLAMA_HOST = "https://127.0.0.1" + _raw_ollama_host[len("https://0.0.0.0"):]
    elif _raw_ollama_host.startswith("0.0.0.0"):
        OLLAMA_HOST = "127.0.0.1" + _raw_ollama_host[len("0.0.0.0"):]
    else:
        OLLAMA_HOST = _raw_ollama_host
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
