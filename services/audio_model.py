from typing import Tuple, Dict


class AudioEmotionModel:
    def __init__(self):
        self.emotion_classes = ["anger", "neutral", "joy", "sadness", "fear"]
        self.model_name = "audio_feature_based_v1"
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        confidence_score = 0.5
        emotion = "neutral"
        
        mfcc_mean = features.get("mfcc_mean", 0)
        energy_mean = features.get("energy_mean", 0)
        amplitude_energy = features.get("amplitude_energy", 0)
        zcr = features.get("zero_crossing_rate", 0.5)
        
        if energy_mean > 0.5 or amplitude_energy > 1000:
            emotion = "anger"
            confidence_score = min(0.9, 0.6 + (energy_mean * 0.3))
        elif zcr > 0.3 and energy_mean > 0.3:
            emotion = "joy"
            confidence_score = min(0.85, 0.55 + (zcr * 0.3))
        elif energy_mean < 0.3 and zcr < 0.2:
            emotion = "sadness"
            confidence_score = min(0.8, 0.5 + (0.3 - energy_mean) * 0.5)
        else:
            emotion = "neutral"
            confidence_score = 0.55
        
        return emotion, max(0.5, min(1.0, confidence_score))