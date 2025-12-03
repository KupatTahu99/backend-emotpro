from utils.text_processing import (
    keyword_based_emotion, 
    extract_text_features,
    validate_and_correct_words,
    preprocess_text,
    get_emotion_explanation
)
from typing import Tuple, Dict


class TextEmotionModel:
    def __init__(self):
        self.emotion_classes = ["anger", "joy", "sadness", "fear", "disgust", "surprise", "trust", "anticipation", "neutral", "horny"]
        self.model_name = "rule_based_v3_comprehensive"
    
    def predict(self, text: str) -> Tuple[str, float, Dict]:
        """
        Prediksi emosi dengan validasi dan koreksi typo yang komprehensif
        """
        # Step 1: Preprocessing
        preprocessed_text = preprocess_text(text)
        
        # Step 2: Validasi dan koreksi typo
        corrected_text, validation_info = validate_and_correct_words(preprocessed_text)
        
        # Step 3: Extract features
        features = extract_text_features(corrected_text)
        
        # Step 4: Deteksi emosi dengan detailed info
        emotion, base_confidence, emotion_matches = keyword_based_emotion(corrected_text)
        
        # Step 5: Adjust confidence berdasarkan features
        if features['word_count'] < 2:
            base_confidence *= 0.6
        elif features['word_count'] > 20:
            base_confidence = min(0.95, base_confidence + 0.1)
        
        # Confidence boost dari punctuation dan caps lock
        punctuation_boost = (features['exclamation_count'] * 0.05) + (features['question_count'] * 0.03)
        caps_boost = features['caps_lock_words'] * 0.02
        final_confidence = min(0.95, base_confidence + punctuation_boost + caps_boost)
        
        # Compile hasil lengkap
        sentiment_scores = {
            "emotion": emotion,
            "confidence": final_confidence,
            "punctuation_boost": round(punctuation_boost, 2),
            "caps_boost": round(caps_boost, 2),
            "explanation": get_emotion_explanation(emotion, emotion_matches.get(emotion, [])),
            "validation_info": validation_info,
            "features": features,
            "original_text": text,
            "corrected_text": corrected_text,
            "corrections_made": validation_info["corrections_made"]
        }
        
        return emotion, final_confidence, sentiment_scores


# Test
if __name__ == "__main__":
    model = TextEmotionModel()
    
    test_cases = [
        "gua lagi pengen berantem",
        "i am very angry and furious",
        "senang banget hari ini",
        "takut takut takut",
        "jijik sekali",
        "terkejut banget"
    ]
    
    for test in test_cases:
        emotion, confidence, scores = model.predict(test)
        print(f"\nTeks: {test}")
        print(f"Emosi: {emotion}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Penjelasan: {scores['explanation']}")
        print(f"Koreksi: {scores['corrections_made']} kata")
        if scores['corrections_made'] > 0:
            print(f"Detail: {scores['validation_info']['correction_details']}")