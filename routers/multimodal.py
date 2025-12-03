from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Optional

# --- IMPORT SERVICE ---
# 1. Import Service Wajah (Yang baru kita buat)
from services.face_model import FaceDetectionService

# 2. Import Service Text (Pastikan file services/text_model.py ada)
# Jika belum ada file services/text_model.py, beritahu saya agar saya buatkan kodenya
try:
    from services.text_model import TextEmotionModel
    text_model = TextEmotionModel()
    print("✅ [Multimodal] Text Model Loaded")
except ImportError:
    text_model = None
    print("⚠️ [Multimodal] Text Model belum ada/gagal load")

from utils.text_processing import preprocess_text

router = APIRouter()

# --- MODELS ---
class TextFaceRequest(BaseModel):
    text: str
    img_data: str  # Gambar Base64 dari Webcam

class MultimodalResponse(BaseModel):
    text_emotion: str
    face_emotion: str
    is_consistent: bool
    final_analysis: str
    details: Dict

# --- ENDPOINTS ---

@router.post("/multimodal/analyze", response_model=MultimodalResponse)
async def analyze_text_and_face(request: TextFaceRequest):
    """
    Menganalisis emosi dari Teks dan Wajah (Kamera) secara bersamaan.
    """
    try:
        # 1. Validasi Input
        if not request.text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if not request.img_data:
            raise HTTPException(status_code=400, detail="Image data cannot be empty")

        # 2. ANALISIS TEKS
        text_emotion = "Unknown"
        if text_model:
            # Preprocessing teks
            clean_text = preprocess_text(request.text)
            # Prediksi (Asumsi method predict me-return emotion, confidence, dll)
            # Sesuaikan dengan output method predict di TextEmotionModel kamu
            prediction = text_model.predict(clean_text)
            
            # Handle jika returnnya tuple atau string langsung
            if isinstance(prediction, tuple):
                text_emotion = prediction[0] # Ambil elemen pertama (label emosi)
            else:
                text_emotion = str(prediction)
        else:
            text_emotion = "Model Text Error"

        # 3. ANALISIS WAJAH
        # Menggunakan face_service yang sudah kita buat sebelumnya
        face_emotion = face_service.predict_emotion(request.img_data)

        # 4. LOGIKA FUSI (PENGGABUNGAN)
        # Cek apakah emosi wajah dan teks sama
        is_consistent = (text_emotion.lower() == face_emotion.lower())
        
        final_analysis = ""
        if is_consistent:
            final_analysis = f"Emosi konsisten! User terlihat dan menulis dengan nada {text_emotion}."
        else:
            final_analysis = f"Terdeteksi ketidakcocokan. Teks bernada '{text_emotion}' tetapi ekspresi wajah menunjukkan '{face_emotion}'."

        # 5. Return Hasil
        return MultimodalResponse(
            text_emotion=text_emotion,
            face_emotion=face_emotion,
            is_consistent=is_consistent,
            final_analysis=final_analysis,
            details={
                "text_input": request.text,
                "face_status": "Detected" if face_emotion != "Wajah tidak terdeteksi" else "Not Detected"
            }
        )

    except Exception as e:
        print(f"Error Multimodal: {e}")
        raise HTTPException(status_code=500, detail=str(e))