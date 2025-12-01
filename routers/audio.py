from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from services.audio_model import AudioEmotionModel
from utils.audio_processing import process_audio_csv
import io

router = APIRouter()
audio_model = AudioEmotionModel()


class AudioResponse(BaseModel):
    emotion: str
    confidence: float
    audio_features: dict
    samples_processed: int


@router.post("/analyze-audio", response_model=AudioResponse)
async def analyze_audio(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        contents = await file.read()
        csv_data = io.BytesIO(contents)
        
        features, raw_data = process_audio_csv(csv_data)
        
        if features is None or len(features) == 0:
            raise HTTPException(status_code=400, detail="No valid audio data in CSV")
        
        emotion, confidence = audio_model.predict(features)
        
        return AudioResponse(
            emotion=emotion,
            confidence=confidence,
            audio_features={
                "mfcc_mean": float(features.get("mfcc_mean", 0)),
                "energy_mean": float(features.get("energy_mean", 0)),
                "zero_crossing_rate": float(features.get("zero_crossing_rate", 0)),
                "spectral_centroid": float(features.get("spectral_centroid", 0))
            },
            samples_processed=raw_data["samples"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))