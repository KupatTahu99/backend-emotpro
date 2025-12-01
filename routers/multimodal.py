from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from services.multimodal_fusion import MultimodalFusion
from utils.text_processing import preprocess_text
from utils.audio_processing import process_audio_csv
import io

router = APIRouter()
multimodal_fusion = MultimodalFusion()


class MultimodalResponse(BaseModel):
    emotion: str
    confidence: float
    text_emotion: str
    text_confidence: float
    audio_emotion: str
    audio_confidence: float
    fusion_weights: dict


@router.post("/analyze-multimodal", response_model=MultimodalResponse)
async def analyze_multimodal(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        processed_text = preprocess_text(text)
        
        contents = await file.read()
        csv_data = io.BytesIO(contents)
        audio_features, _ = process_audio_csv(csv_data)
        
        if audio_features is None or len(audio_features) == 0:
            raise HTTPException(status_code=400, detail="No valid audio data in CSV")
        
        result = multimodal_fusion.fuse(
            text=processed_text,
            audio_features=audio_features
        )
        
        return MultimodalResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))