from services.text_model import TextEmotionModel
from services.audio_model import AudioEmotionModel
from typing import Dict


class MultimodalFusion:
    def __init__(self, text_weight: float = 0.5, audio_weight: float = 0.5):
        self.text_model = TextEmotionModel()
        self.audio_model = AudioEmotionModel()
        self.text_weight = text_weight
        self.audio_weight = audio_weight
    
    def fuse(self, text: str, audio_features: Dict) -> Dict:
        text_emotion, text_confidence, _ = self.text_model.predict(text)
        audio_emotion, audio_confidence = self.audio_model.predict(audio_features)
        
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
            }
        }
    
    def _late_fusion(self, text_emotion: str, text_conf: float,
                     audio_emotion: str, audio_conf: float) -> tuple:
        if text_emotion == audio_emotion:
            fused_confidence = (
                text_conf * self.text_weight +
                audio_conf * self.audio_weight
            )
            fused_confidence = min(1.0, fused_confidence + 0.1)
            return text_emotion, fused_confidence
        
        text_score = text_conf * self.text_weight
        audio_score = audio_conf * self.audio_weight
        
        if text_score >= audio_score:
            fused_emotion = text_emotion
            fused_confidence = text_score
        else:
            fused_emotion = audio_emotion
            fused_confidence = audio_score
        
        return fused_emotion, min(0.95, fused_confidence)