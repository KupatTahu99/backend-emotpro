from utils.text_processing import keyword_based_emotion, extract_text_features
from typing import Tuple, Dict

class TextEmotionModel:
    """
    Text emotion detection model
    Uses rule-based approach with keyword matching
    """
    
    def __init__(self):
        self.emotion_classes = ["anger", "joy", "sadness", "fear", "neutral"]
        self.model_name = "rule_based_v1"
    
    def predict(self, text: str) -> Tuple[str, float, Dict]:
        """
        Predict emotion from text
        Returns: (emotion, confidence, sentiment_scores)
        """
        # Get features
        features = extract_text_features(text)
        
        # Get emotion from keywords
        emotion, base_confidence = keyword_based_emotion(text)
        
        # Adjust confidence based on text length
        if features['word_count'] < 3:
            base_confidence *= 0.7
        elif features['word_count'] > 20:
            base_confidence = min(0.95, base_confidence + 0.1)
        
        # Create sentiment scores
        sentiment_scores = {
            emotion: min(1.0, base_confidence + 0.05),
            "neutral": 1 - base_confidence,
            "confidence_boost": features['exclamation_count'] * 0.05
        }
        
        return emotion, base_confidence, sentiment_scores