from services.text_model import TextEmotionModel
from services.face_model import FaceDetectionService
from typing import Dict, Tuple


class MultimodalFusion:
    """
    Fuse text dan audio emotion signals untuk lebih akurat emotion detection
    """
    
    def __init__(self, text_weight: float = 0.5, audio_weight: float = 0.5):
        self.text_model = TextEmotionModel()
        self.audio_model = FaceDetectionService()
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        
        # Emotion similarity matrix
        self.emotion_similarity = self._create_emotion_similarity_matrix()
    
    def fuse(self, text: str, audio_features: Dict) -> Dict:
        """
        Fuse text dan audio analysis
        """
        # Text prediction
        text_emotion, text_confidence, text_scores = self.text_model.predict(text)
        
        # Audio prediction
        audio_emotion, audio_confidence, audio_scores = self.audio_model.predict(audio_features)
        
        # Late fusion
        fused_emotion, fused_confidence = self._late_fusion(
            text_emotion, text_confidence,
            audio_emotion, audio_confidence
        )
        
        return {
            "emotion": fused_emotion,
            "confidence": fused_confidence,
            "text_emotion": text_emotion,
            "text_confidence": text_confidence,
            "audio_emotion": audio_emotion,
            "audio_confidence": audio_confidence,
            "fusion_weights": {
                "text_weight": self.text_weight,
                "audio_weight": self.audio_weight
            },
            "text_emotion_details": text_scores,
            "audio_emotion_details": audio_scores
        }
    
    def _late_fusion(
        self,
        text_emotion: str,
        text_conf: float,
        audio_emotion: str,
        audio_conf: float
    ) -> Tuple[str, float]:
        """
        Late fusion: Combine scores dari kedua modality
        """
        
        # Jika kedua modality agree
        if text_emotion == audio_emotion:
            fused_confidence = (
                text_conf * self.text_weight +
                audio_conf * self.audio_weight
            )
            fused_confidence = min(1.0, fused_confidence + 0.1)  # Boost confidence
            return text_emotion, fused_confidence
        
        # Jika tidak agree, hitung berdasarkan weighted scores
        text_score = text_conf * self.text_weight
        audio_score = audio_conf * self.audio_weight
        
        # Choose emotion dengan higher weighted score
        if text_score >= audio_score:
            fused_emotion = text_emotion
            fused_confidence = text_score
        else:
            fused_emotion = audio_emotion
            fused_confidence = audio_score
        
        # Apply similarity boost jika emotions related
        similarity = self.emotion_similarity.get(
            (text_emotion, audio_emotion),
            self.emotion_similarity.get((audio_emotion, text_emotion), 0)
        )
        
        if similarity > 0:
            fused_confidence = min(1.0, fused_confidence + (similarity * 0.1))
        
        return fused_emotion, fused_confidence
    
    @staticmethod
    def _create_emotion_similarity_matrix() -> Dict:
        """
        Create matrix menunjukkan seberapa similar emotions
        """
        return {
            ("anger", "disgust"): 0.8,
            ("anger", "fear"): 0.6,
            ("joy", "trust"): 0.9,
            ("joy", "anticipation"): 0.8,
            ("sadness", "fear"): 0.7,
            ("sadness", "disgust"): 0.5,
            ("fear", "anticipation"): 0.4,
            ("trust", "anticipation"): 0.7,
            ("surprise", "anticipation"): 0.6,
            ("neutral", "trust"): 0.5,
        }