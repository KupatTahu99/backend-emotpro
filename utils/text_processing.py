import re
from typing import Dict


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text_features(text: str) -> Dict:
    features = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "punctuation_count": sum(1 for c in text if c in "!?.,-"),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
    }
    return features


def keyword_based_emotion(text: str) -> tuple:
    text_lower = text.lower()
    
    anger_keywords = ["angry", "furious", "rage", "hate", "worst", "terrible", "awful"]
    joy_keywords = ["happy", "joy", "love", "great", "wonderful", "amazing", "excellent"]
    sad_keywords = ["sad", "depressed", "unhappy", "terrible", "horrible", "bad"]
    fear_keywords = ["scared", "afraid", "worried", "anxious", "fear", "nervous"]
    
    emotions = {
        "anger": sum(1 for kw in anger_keywords if kw in text_lower),
        "joy": sum(1 for kw in joy_keywords if kw in text_lower),
        "sadness": sum(1 for kw in sad_keywords if kw in text_lower),
        "fear": sum(1 for kw in fear_keywords if kw in text_lower),
    }
    
    if not any(emotions.values()):
        return "neutral", 0.5
    
    max_emotion = max(emotions, key=emotions.get)
    confidence = min(0.95, 0.6 + (emotions[max_emotion] * 0.1))
    
    return max_emotion, confidence