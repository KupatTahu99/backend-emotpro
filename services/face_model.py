import cv2
import numpy as np
from typing import List, Tuple

class FaceDetectionService:
    def __init__(self, cascade_path='models/haarcascade_frontalface_default.xml'):
        """
        Initialize face detection using Haar Cascade
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Jika file tidak ditemukan, gunakan default OpenCV
        if self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_face(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face region from image
        """
        x, y, w, h = face_coords
        face_img = image[y:y+h, x:x+w]
        return face_img
    
face_service = FaceDetectionService()