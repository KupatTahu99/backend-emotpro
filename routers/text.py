from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.text_model import TextEmotionModel
from utils.text_processing import preprocess_text

router = APIRouter()
text_model = TextEmotionModel()


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    emotion: str
    confidence: float
    sentiment_scores: dict
    processed_text: str


@router.post("/analyze-text", response_model=TextResponse)
async def analyze_text(request: TextRequest):
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        processed = preprocess_text(request.text)
        emotion, confidence, scores = text_model.predict(processed)
        
        return TextResponse(
            emotion=emotion,
            confidence=confidence,
            sentiment_scores=scores,
            processed_text=processed
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))