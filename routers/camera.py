from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from pydantic import BaseModel
from typing import Dict, List
import cv2
import numpy as np
import base64
import io
from PIL import Image

from services.emotion_cnn_model import EmotionCNNModel

router = APIRouter()
emotion_model = EmotionCNNModel(model_path='models/emotion_model.h5')


class CameraFrameRequest(BaseModel):
    frame: str  # base64 encoded image


class CameraAnalysisResponse(BaseModel):
    success: bool
    faces_detected: int
    faces: List[Dict]


@router.post("/camera/analyze-frame", response_model=CameraAnalysisResponse)
async def analyze_camera_frame(data: CameraFrameRequest):
    """
    Analyze single frame dari webcam
    
    Request:
    {
        "frame": "base64_encoded_image_data"
    }
    
    Response:
    {
        "success": true,
        "faces_detected": 1,
        "faces": [
            {
                "id": 0,
                "coordinates": {"x": 100, "y": 120, "width": 80, "height": 100},
                "emotion": "Senang",
                "emotion_en": "Happy",
                "emotion_id": 3,
                "confidence": 0.95
            }
        ]
    }
    """
    try:
        # Decode base64
        if data.frame.startswith("data:image"):
            frame_data = data.frame.split(",")[1]
        else:
            frame_data = data.frame
        
        image_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Analyze frame
        result = emotion_model.analyze_frame(frame)
        
        return CameraAnalysisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")


@router.post("/camera/analyze-with-annotation")
async def analyze_frame_with_annotation(data: CameraFrameRequest):
    """
    Analyze frame dan return annotated frame dengan boxes + emotion labels
    """
    try:
        # Decode base64
        if data.frame.startswith("data:image"):
            frame_data = data.frame.split(",")[1]
        else:
            frame_data = data.frame
        
        image_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Analyze
        analysis = emotion_model.analyze_frame(frame)
        
        # Draw predictions
        annotated = emotion_model.draw_predictions(frame, analysis)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_base64 = base64.b64encode(buffer).decode()
        
        return {
            "analysis": analysis,
            "annotated_frame": f"data:image/jpeg;base64,{frame_base64}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/camera/batch-analyze-frames")
async def batch_analyze_frames(frames: List[CameraFrameRequest]):
    """
    Analyze multiple frames sekaligus
    Max 10 frames
    """
    try:
        if len(frames) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 frames allowed")
        
        results = []
        emotion_stats = {}
        
        for frame_req in frames:
            try:
                if frame_req.frame.startswith("data:image"):
                    frame_data = frame_req.frame.split(",")[1]
                else:
                    frame_data = frame_req.frame
                
                image_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    analysis = emotion_model.analyze_frame(frame)
                    results.append(analysis)
                    
                    # Count emotions
                    for face in analysis["faces"]:
                        emotion = face["emotion"]
                        emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
            
            except:
                continue
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid frames processed")
        
        return {
            "total_frames": len(results),
            "results": results,
            "emotion_distribution": emotion_stats
        }
    
    except HTTPException:
        raise
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/camera/analyze-image-file")
async def analyze_image_file(file: UploadFile = File(...)):
    """
    Upload image file dan analyze
    """
    try:
        if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        result = emotion_model.analyze_frame(frame)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera/emotions")
async def get_emotions():
    """
    Get list of available emotions
    """
    return {
        "emotions": emotion_model.emotion_labels,
        "emotions_en": emotion_model.emotion_labels_en
    }


@router.get("/camera/emotion-colors")
async def get_emotion_colors():
    """
    Get colors untuk setiap emotion (untuk frontend)
    """
    return {
        "0": {"emotion": "Marah", "color": "#FF0000"},
        "1": {"emotion": "Jijik", "color": "#FFA500"},
        "2": {"emotion": "Takut", "color": "#0000FF"},
        "3": {"emotion": "Senang", "color": "#00FF00"},
        "4": {"emotion": "Netral", "color": "#808080"},
        "5": {"emotion": "Sedih", "color": "#800080"},
        "6": {"emotion": "Terkejut", "color": "#FFFF00"}
    }