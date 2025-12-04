from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64

# Import Service
from services.face_model import FaceDetectionService
from services.emotion_cnn_model import EmotionCNNModel # Class yang baru kita buat

router = APIRouter(prefix="/vision", tags=["Vision"])

# Initialize services
face_detector = FaceDetectionService()
emotion_recognizer = EmotionCNNModel()

@router.post("/detect-emotion")
async def detect_emotion(file: UploadFile = File(...)):
    """
    Menerima gambar upload, mendeteksi wajah, dan memprediksi emosi menggunakan CNN.
    """
    try:
        # 1. Baca Gambar dari Upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 2. Deteksi Lokasi Wajah
        faces = face_detector.detect_faces(image)
        
        if len(faces) == 0:
            return JSONResponse(content={
                "success": False,
                "message": "No faces detected",
                "faces": []
            })
        
        results = []
        
        # 3. Loop setiap wajah yang ketemu
        for idx, (x, y, w, h) in enumerate(faces):
            # Extract/Crop bagian wajah saja
            # (Pastikan method extract_face ada di FaceDetectionService Anda)
            face_img = image[y:y+h, x:x+w]
            
            # 4. Prediksi Emosi menggunakan CNN Model
            emotion_result = emotion_recognizer.predict_emotion(face_img)
            
            # (Opsional) Gambar kotak di foto asli untuk dikirim balik
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{emotion_result['emotion']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            results.append({
                "face_id": idx,
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "emotion": emotion_result['emotion'],
                "confidence": emotion_result['confidence'],
                "all_scores": emotion_result['all_probabilities']
            })
        
        # Convert gambar hasil anotasi ke Base64 (untuk preview di frontend)
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "success": True,
            "message": f"Detected {len(faces)} face(s)",
            "faces": results,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        })
        
    except Exception as e:
        print(f"Error Vision API: {e}")
        raise HTTPException(status_code=500, detail=str(e))