import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

import src.config as config
from src.engines.stt_engine import STTEngine
from src.engines.llm_engine import LLMEngine
from src.engines.tts_engine import TTSEngine
from src.engines.llm_engine import OllamaUnavailableError
from src.utils.audio_tools import preprocess_for_stt, preprocess_for_tts

# 1. Initialize API and AI Engines
app = FastAPI()

print("Loading Engines... (Models will be loaded from D:\\VoxSync_Cache)")
stt = STTEngine()
llm = LLMEngine()
tts = TTSEngine()
print("All engines loaded successfully!")

# Ensure data directories exist
DATA_DIR = Path("data")
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. The Chat API Endpoint
@app.post("/api/chat")
async def chat_endpoint(
    mic_audio: UploadFile = File(...), 
    ref_audio: UploadFile = File(...)
):
    # Save uploaded files temporarily
    mic_filename = Path(mic_audio.filename or "mic_input").name
    ref_filename = Path(ref_audio.filename or "ref_audio").name
    mic_path = TEMP_DIR / mic_filename
    ref_path = TEMP_DIR / ref_filename
    
    with open(mic_path, "wb") as buffer:
        shutil.copyfileobj(mic_audio.file, buffer)
    with open(ref_path, "wb") as buffer:
        shutil.copyfileobj(ref_audio.file, buffer)

    # Process Audio -> STT -> LLM -> TTS
    try:
        clean_mic = preprocess_for_stt(str(mic_path))
        clean_ref = preprocess_for_tts(str(ref_path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    user_text = stt.transcribe(clean_mic)
    try:
        ai_text = llm.chat(user_text)
    except OllamaUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    output_audio_path = tts.generate(ai_text, clean_ref)

    # Return the AI text and the URL to fetch the generated audio
    return {
        "user_text": user_text,
        "ai_text": ai_text,
        "audio_url": f"/audio/{Path(output_audio_path).name}"
    }

# 3. Endpoint to serve the generated audio files
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = OUTPUT_DIR / filename
    return FileResponse(file_path)

# 4. Mount the UI folder to serve HTML/JS/CSS at the root URL
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)