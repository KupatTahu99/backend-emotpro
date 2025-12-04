from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import cv2
import numpy as np
import base64

# --- IMPORT SERVICES ---
from services.face_model import FaceDetectionService
from services.emotion_cnn_model import EmotionCNNModel
from services.text_model import TextEmotionModel
from utils.text_processing import preprocess_text, validate_and_correct_words

router = APIRouter()

# --- INITIALIZE MODELS ---
try:
    text_model = TextEmotionModel()
    print("✅ [Multimodal] Text model loaded")
except Exception as e:
    text_model = None
    print(f"❌ Text model error: {e}")

try:
    # Initialize Face Detector (Haar) & Emotion Recognizer (CNN)
    face_detector = FaceDetectionService()
    emotion_recognizer = EmotionCNNModel(model_path='models/model.h5') # Pastikan nama file benar
    print("✅ [Multimodal] Camera models loaded")
except Exception as e:
    face_detector = None
    emotion_recognizer = None
    print(f"❌ Camera model error: {e}")


# --- PYDANTIC MODELS ---
class MultimodalRequest(BaseModel):
    text: str
    img_data: str # Menggunakan nama 'img_data' agar konsisten dengan frontend

class MultimodalResponse(BaseModel):
    success: bool
    text_emotion: str
    text_confidence: float
    camera_emotion: str
    camera_confidence: float
    is_consistent: bool
    consistency_score: float
    final_analysis: str
    recommendation: str
    details: Dict

# --- HELPER FUNCTIONS ---
def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Decode error: {e}")
        return None

def calculate_consistency(text_emo: str, cam_emo: str, text_conf: float, cam_conf: float):
    # Mapping sederhana sinonim emosi (Indonesian/English)
    synonyms = {
        "marah": ["marah", "angry", "anger", "furious"],
        "senang": ["senang", "happy", "joy", "gembira"],
        "sedih": ["sedih", "sad", "sadness", "depressed"],
        "takut": ["takut", "fear", "fearful", "scared"],
        "jijik": ["jijik", "disgust", "disgusted"],
        "terkejut": ["terkejut", "surprise", "surprised"],
        "netral": ["netral", "neutral"]
    }

    t_lower = text_emo.lower()
    c_lower = cam_emo.lower()
    
    match = False
    # Cek direct match
    if t_lower == c_lower:
        match = True
    else:
        # Cek via sinonim
        for key, vals in synonyms.items():
            if t_lower in vals and c_lower in vals:
                match = True
                break
    
    base_score = 1.0 if match else 0.2
    # Consistency score dipengaruhi confidence model
    final_score = base_score * ((text_conf + cam_conf) / 2)
    
    return match, final_score

# --- ENDPOINT ---
# URL Akhir: http://localhost:8000/api/multimodal/analyze
@router.post("/analyze", response_model=MultimodalResponse)
async def analyze_multimodal(request: MultimodalRequest):
    try:
        # 1. Validasi Input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text empty")
        if not request.img_data:
            raise HTTPException(status_code=400, detail="Image empty")

        # 2. ANALISIS TEKS
        text_res = "Netral"
        text_conf = 0.0
        corrections = 0
        
        if text_model:
            try:
                # Preprocess -> Correct -> Predict
                clean_text = preprocess_text(request.text)
                corrected_text, correction_info = validate_and_correct_words(clean_text)
                corrections = correction_info.get("corrections_made", 0)
                
                # Predict returns tuple: (emotion, confidence, details)
                prediction = text_model.predict(corrected_text)
                text_res = prediction[0]
                text_conf = prediction[1]
            except Exception as e:
                print(f"Text Analysis Error: {e}")
                text_res = "Error"

        # 3. ANALISIS KAMERA (PIPELINE BARU)
        cam_res = "Wajah tidak terdeteksi"
        cam_conf = 0.0
        faces_count = 0

        if face_detector and emotion_recognizer:
            try:
                # A. Decode Image
                image = decode_base64_image(request.img_data)
                if image is None:
                    raise ValueError("Invalid Image Data")

                # B. Detect Faces
                faces = face_detector.detect_faces(image)
                faces_count = len(faces)

                if faces_count > 0:
                    # C. Extract Face (Ambil wajah pertama)
                    x, y, w, h = faces[0]
                    # Gunakan method extract_face dari service Anda
                    face_img = face_detector.extract_face(image, (x, y, w, h))

                    # D. Predict Emotion (Pakai CNN)
                    result = emotion_recognizer.predict_emotion(face_img)
                    cam_res = result['emotion']
                    cam_conf = result['confidence']
            except Exception as e:
                print(f"Camera Analysis Error: {e}")
                cam_res = "Error"

        # 4. FUSION & CONSISTENCY
        is_consistent, consistency_score = calculate_consistency(text_res, cam_res, text_conf, cam_conf)

        # 5. GENERATE RECOMMENDATION
        final_analysis = ""
        recommendation = ""

        if cam_res == "Wajah tidak terdeteksi":
            final_analysis = f"Hanya analisis teks berhasil: {text_res}."
            recommendation = "Pastikan wajah terlihat jelas di kamera."
            is_consistent = False
        elif is_consistent:
            final_analysis = f"✅ Konsisten: Teks dan Wajah sama-sama menunjukkan {text_res}."
            recommendation = "Analisis valid. User mengekspresikan emosi yang jujur."
        else:
            final_analysis = f"⚠️ Inkonsisten: Teks '{text_res}' tapi Wajah '{cam_res}'."
            if text_res.lower() == "senang" and cam_res.lower() in ["marah", "sedih"]:
                recommendation = "Kemungkinan Sarkasme atau user menyembunyikan perasaan."
            else:
                recommendation = "Terdeteksi perbedaan ekspresi. Perlu verifikasi manual."

        return MultimodalResponse(
            success=True,
            text_emotion=text_res,
            text_confidence=text_conf,
            camera_emotion=cam_res,
            camera_confidence=cam_conf,
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            final_analysis=final_analysis,
            recommendation=recommendation,
            details={
                "input_text": request.text,
                "corrections": corrections,
                "faces_detected": faces_count
            }
        )

    except Exception as e:
        print(f"Global Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))