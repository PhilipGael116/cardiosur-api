import io
import numpy as np
import soundfile as sf
import librosa

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.preprocess import preprocess
from services.model import predict
from services.graph import generate_graph

router = APIRouter()


def read_audio(file_bytes: bytes) -> tuple:
    """
    Read audio file bytes into a numpy array.
    Supports WAV, WebM, MP3 and most common formats.
    Returns (audio_array, sample_rate)
    """
    try:
        # try soundfile first — handles WAV cleanly
        audio, sr = sf.read(io.BytesIO(file_bytes))
    except Exception:
        try:
            # fallback to librosa — handles WebM, MP3 etc
            audio, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
            return audio, sr
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read audio file: {str(e)}"
            )

    # convert stereo to mono by averaging channels
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # ensure float32
    audio = audio.astype(np.float32)

    return audio, sr


@router.post("/predict")
async def predict_heart_sound(file: UploadFile = File(...)):
    """
    Main endpoint.
    Receives a WAV/WebM audio file, preprocesses it,
    runs AI inference and returns the result.

    Request:
        POST /api/predict
        Body: multipart/form-data with field "file"

    Response:
        {
            "result": "normal" | "abnormal",
            "bpm": 72,
            "confidence": 0.94,
            "chunks_analyzed": 30,
            "raw_score": 0.043,
            "graph": "<base64 PNG string>"
        }
    """

    # 1 — validate file type
    allowed_types = ["audio/wav", "audio/webm", "audio/mp3", "audio/mpeg", "audio/ogg", "application/octet-stream"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an audio file."
        )

    # 2 — read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {str(e)}")

    # 3 — check file is not empty
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 4 — decode audio
    audio, original_sr = read_audio(file_bytes)

    # 5 — check audio is long enough (need at least 2 seconds)
    duration_seconds = len(audio) / original_sr
    if duration_seconds < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too short: {duration_seconds:.1f}s. Minimum is 2 seconds."
        )

    # 6 — preprocess: resample → filter → standardize → chunk → bpm
    try:
        chunks, bpm = preprocess(audio, original_sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    # 7 — check we got valid chunks
    if len(chunks) == 0:
        raise HTTPException(
            status_code=400,
            detail="Audio produced no valid chunks after preprocessing. Try recording for longer."
        )

    # 8 — run model inference
    try:
        prediction = predict(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 9 — generate analysis graph
    try:
        graph_b64 = generate_graph(audio, original_sr, prediction["result"], bpm)
    except Exception as e:
        graph_b64 = None  # graph failure should not break the prediction

    # 10 — return result
    return {
        "result": prediction["result"],
        "bpm": bpm,
        "confidence": prediction["confidence"],
        "chunks_analyzed": prediction["chunks_analyzed"],
        "raw_score": prediction.get("raw_score"),
        "graph": graph_b64
    }