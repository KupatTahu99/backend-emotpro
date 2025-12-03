import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import io


# Audio feature ranges untuk setiap emosi berdasarkan EMOTION_VOCABULARY
AUDIO_EMOTION_PROFILES = {
    "anger": {
        "amplitude_range": (0.6, 1.0),  # Amplitude tinggi
        "energy_range": (0.7, 1.0),     # Energy tinggi
        "mfcc_range": (0.5, 0.8),       # MFCC sedang-tinggi
        "zero_crossing_rate": (0.4, 0.7),  # ZCR sedang
        "spectral_centroid": (0.6, 0.9),   # Frekuensi tinggi
        "pitch_variation": (0.5, 0.8),  # Variasi pitch sedang-tinggi
        "tempo": (0.7, 1.0),            # Tempo cepat
        "characteristics": ["loud", "intense", "harsh", "fast"]
    },
    
    "joy": {
        "amplitude_range": (0.6, 0.9),  # Amplitude sedang-tinggi
        "energy_range": (0.6, 0.9),     # Energy sedang-tinggi
        "mfcc_range": (0.5, 0.8),       # MFCC sedang-tinggi
        "zero_crossing_rate": (0.3, 0.6),  # ZCR sedang
        "spectral_centroid": (0.5, 0.8),   # Frekuensi sedang-tinggi
        "pitch_variation": (0.6, 0.9),  # Variasi pitch tinggi (dinamis)
        "tempo": (0.6, 0.8),            # Tempo sedang-cepat
        "characteristics": ["bright", "dynamic", "warm", "melodic"]
    },
    
    "sadness": {
        "amplitude_range": (0.2, 0.5),  # Amplitude rendah
        "energy_range": (0.2, 0.5),     # Energy rendah
        "mfcc_range": (0.3, 0.6),       # MFCC rendah-sedang
        "zero_crossing_rate": (0.2, 0.5),  # ZCR rendah
        "spectral_centroid": (0.2, 0.5),   # Frekuensi rendah
        "pitch_variation": (0.2, 0.4),  # Variasi pitch rendah (monoton)
        "tempo": (0.2, 0.4),            # Tempo lambat
        "characteristics": ["soft", "low", "slow", "smooth"]
    },
    
    "fear": {
        "amplitude_range": (0.4, 0.8),  # Amplitude sedang-tinggi
        "energy_range": (0.4, 0.8),     # Energy sedang-tinggi
        "mfcc_range": (0.4, 0.7),       # MFCC sedang
        "zero_crossing_rate": (0.5, 0.8),  # ZCR tinggi (banyak noise)
        "spectral_centroid": (0.4, 0.7),   # Frekuensi sedang
        "pitch_variation": (0.6, 0.9),  # Variasi pitch tinggi (unstable)
        "tempo": (0.5, 0.8),            # Tempo tidak stabil
        "characteristics": ["tense", "unstable", "variable", "nervous"]
    },
    
    "disgust": {
        "amplitude_range": (0.5, 0.8),  # Amplitude sedang-tinggi
        "energy_range": (0.5, 0.8),     # Energy sedang-tinggi
        "mfcc_range": (0.4, 0.7),       # MFCC sedang
        "zero_crossing_rate": (0.4, 0.7),  # ZCR sedang
        "spectral_centroid": (0.3, 0.6),   # Frekuensi rendah-sedang
        "pitch_variation": (0.3, 0.6),  # Variasi pitch rendah
        "tempo": (0.4, 0.6),            # Tempo lambat-sedang
        "characteristics": ["harsh", "heavy", "disgusted", "rejection"]
    },
    
    "surprise": {
        "amplitude_range": (0.6, 0.95), # Amplitude tinggi (sudden)
        "energy_range": (0.6, 0.95),    # Energy tinggi
        "mfcc_range": (0.5, 0.85),      # MFCC tinggi
        "zero_crossing_rate": (0.4, 0.7),  # ZCR sedang
        "spectral_centroid": (0.6, 0.9),   # Frekuensi tinggi
        "pitch_variation": (0.7, 0.95), # Variasi pitch sangat tinggi (spike)
        "tempo": (0.6, 0.9),            # Tempo cepat
        "characteristics": ["sudden", "sharp", "spike", "unexpected"]
    },
    
    "trust": {
        "amplitude_range": (0.5, 0.75), # Amplitude sedang
        "energy_range": (0.5, 0.75),    # Energy sedang
        "mfcc_range": (0.5, 0.75),      # MFCC sedang
        "zero_crossing_rate": (0.3, 0.6),  # ZCR rendah-sedang
        "spectral_centroid": (0.5, 0.75),  # Frekuensi sedang
        "pitch_variation": (0.3, 0.5),  # Variasi pitch rendah (stable)
        "tempo": (0.5, 0.7),            # Tempo stabil
        "characteristics": ["calm", "stable", "confident", "steady"]
    },
    
    "anticipation": {
        "amplitude_range": (0.5, 0.85), # Amplitude sedang-tinggi
        "energy_range": (0.5, 0.85),    # Energy sedang-tinggi
        "mfcc_range": (0.5, 0.8),       # MFCC sedang-tinggi
        "zero_crossing_rate": (0.35, 0.65),  # ZCR sedang
        "spectral_centroid": (0.55, 0.85),  # Frekuensi sedang-tinggi
        "pitch_variation": (0.4, 0.7),  # Variasi pitch sedang
        "tempo": (0.6, 0.85),           # Tempo cepat-sedang
        "characteristics": ["building", "expectant", "progressive", "energetic"]
    },
    
    "horny": {
        "amplitude_range": (0.6, 0.9),  # Amplitude sedang-tinggi
        "energy_range": (0.6, 0.9),     # Energy sedang-tinggi
        "mfcc_range": (0.5, 0.8),       # MFCC sedang-tinggi
        "zero_crossing_rate": (0.3, 0.6),  # ZCR sedang
        "spectral_centroid": (0.4, 0.7),   # Frekuensi sedang
        "pitch_variation": (0.6, 0.9),  # Variasi pitch tinggi (sensual)
        "tempo": (0.5, 0.8),            # Tempo sedang
        "characteristics": ["sensual", "rhythmic", "warm", "intimate"]
    },
    
    "neutral": {
        "amplitude_range": (0.3, 0.7),  # Amplitude sedang
        "energy_range": (0.3, 0.7),     # Energy sedang
        "mfcc_range": (0.4, 0.7),       # MFCC sedang
        "zero_crossing_rate": (0.3, 0.6),  # ZCR sedang
        "spectral_centroid": (0.4, 0.7),   # Frekuensi sedang
        "pitch_variation": (0.3, 0.5),  # Variasi pitch rendah
        "tempo": (0.4, 0.6),            # Tempo sedang-lambat
        "characteristics": ["balanced", "moderate", "flat", "robotic"]
    }
}


def normalize_feature(value: float, min_val: float = 0, max_val: float = 1) -> float:
    """Normalize nilai feature ke range 0-1"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return max(0, min(1, normalized))


def process_audio_csv(csv_file: io.BytesIO) -> Tuple[Optional[Dict], Dict]:
    """
    Process audio CSV dan extract features yang sesuai dengan emotion vocab
    """
    try:
        df = pd.read_csv(csv_file)
        
        if df.empty:
            return None, {"samples": 0}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return None, {"samples": 0}
        
        features = {}
        raw_features = {}  # Untuk tracking raw values
        
        # Amplitude Analysis
        if 'amplitude' in df.columns:
            amplitude = df['amplitude'].values
            amplitude_mean = float(np.mean(amplitude))
            amplitude_max = float(np.max(amplitude))
            
            raw_features['amplitude_mean'] = amplitude_mean
            raw_features['amplitude_max'] = amplitude_max
            raw_features['amplitude_std'] = float(np.std(amplitude))
            
            # Normalize amplitude
            features['amplitude_mean'] = normalize_feature(amplitude_mean, 0, 1)
            features['amplitude_energy'] = normalize_feature(
                float(np.sum(amplitude ** 2)) / len(amplitude), 0, 1
            )
        
        # MFCC Analysis (Mel-Frequency Cepstral Coefficients)
        mfcc_cols = [col for col in numeric_cols if 'mfcc' in col.lower()]
        if mfcc_cols:
            mfcc_data = df[mfcc_cols].values
            mfcc_mean = float(np.mean(mfcc_data))
            mfcc_std = float(np.std(mfcc_data))
            
            raw_features['mfcc_mean'] = mfcc_mean
            raw_features['mfcc_std'] = mfcc_std
            raw_features['mfcc_coefficients'] = int(len(mfcc_cols))
            
            # Normalize MFCC
            features['mfcc_mean'] = normalize_feature(mfcc_mean, -20, 20)
            features['mfcc_std'] = normalize_feature(mfcc_std, 0, 10)
        
        # Energy Analysis
        if 'energy' in df.columns:
            energy = df['energy'].values
            energy_mean = float(np.mean(energy))
            
            raw_features['energy_mean'] = energy_mean
            raw_features['energy_std'] = float(np.std(energy))
            
            features['energy_mean'] = normalize_feature(energy_mean, 0, 1)
        
        # Zero Crossing Rate (untuk mendeteksi fricatives/noise)
        if 'zero_crossing_rate' in df.columns:
            zcr = df['zero_crossing_rate'].values
            zcr_mean = float(np.mean(zcr))
            
            raw_features['zero_crossing_rate'] = zcr_mean
            features['zero_crossing_rate'] = normalize_feature(zcr_mean, 0, 0.5)
        
        # Spectral Centroid (center of mass of spectrum)
        if 'spectral_centroid' in df.columns:
            sc = df['spectral_centroid'].values
            sc_mean = float(np.mean(sc))
            
            raw_features['spectral_centroid'] = sc_mean
            features['spectral_centroid'] = normalize_feature(sc_mean, 0, 8000)
        
        # Pitch Analysis (jika ada)
        if 'pitch' in df.columns:
            pitch = df['pitch'].values
            pitch_mean = float(np.mean(pitch[pitch > 0]))  # Exclude unvoiced
            pitch_std = float(np.std(pitch[pitch > 0]))
            
            raw_features['pitch_mean'] = pitch_mean
            raw_features['pitch_std'] = pitch_std
            
            features['pitch_variation'] = normalize_feature(pitch_std, 0, 200)
        
        # Tempo/Rate Analysis
        if 'rate' in df.columns or 'tempo' in df.columns:
            col_name = 'tempo' if 'tempo' in df.columns else 'rate'
            tempo = df[col_name].values
            tempo_mean = float(np.mean(tempo))
            
            raw_features['tempo'] = tempo_mean
            features['tempo'] = normalize_feature(tempo_mean, 0.5, 2.0)
        
        # Jika fitur minimal tidak ada, gunakan numeric cols pertama
        if len(features) < 3:
            for col in numeric_cols[:5]:
                col_name = col.lower()
                col_data = df[col].values
                col_mean = float(np.mean(col_data))
                col_max = float(np.max(col_data))
                
                raw_features[col_name] = col_mean
                features[col_name] = normalize_feature(col_mean, 0, col_max if col_max > 0 else 1)
        
        metadata = {
            "samples": len(df),
            "total_columns": len(df.columns),
            "numeric_features": len(numeric_cols),
            "raw_features": raw_features
        }
        
        return features, metadata
        
    except pd.errors.ParserError:
        return None, {"samples": 0, "error": "Invalid CSV format"}
    except Exception as e:
        return None, {"samples": 0, "error": str(e)}


def calculate_emotion_score(
    audio_features: Dict[str, float],
    emotion: str
) -> Tuple[float, Dict]:
    """
    Hitung skor emosi berdasarkan audio features dan emotion profile
    
    Returns:
        Tuple: (confidence_score, feature_analysis)
    """
    if emotion not in AUDIO_EMOTION_PROFILES:
        return 0.5, {}
    
    profile = AUDIO_EMOTION_PROFILES[emotion]
    scores = {}
    
    # Check amplitude
    if 'amplitude_mean' in audio_features and 'amplitude_range' in profile:
        amp_range = profile['amplitude_range']
        amp_value = audio_features['amplitude_mean']
        if amp_range[0] <= amp_value <= amp_range[1]:
            scores['amplitude'] = 1.0
        else:
            # Calculate distance from range
            if amp_value < amp_range[0]:
                distance = (amp_range[0] - amp_value) / amp_range[0]
            else:
                distance = (amp_value - amp_range[1]) / (1 - amp_range[1])
            scores['amplitude'] = max(0, 1 - distance)
    
    # Check energy
    if 'energy_mean' in audio_features and 'energy_range' in profile:
        energy_range = profile['energy_range']
        energy_value = audio_features['energy_mean']
        if energy_range[0] <= energy_value <= energy_range[1]:
            scores['energy'] = 1.0
        else:
            if energy_value < energy_range[0]:
                distance = (energy_range[0] - energy_value) / energy_range[0]
            else:
                distance = (energy_value - energy_range[1]) / (1 - energy_range[1])
            scores['energy'] = max(0, 1 - distance)
    
    # Check MFCC
    if 'mfcc_mean' in audio_features and 'mfcc_range' in profile:
        mfcc_range = profile['mfcc_range']
        mfcc_value = audio_features['mfcc_mean']
        if mfcc_range[0] <= mfcc_value <= mfcc_range[1]:
            scores['mfcc'] = 1.0
        else:
            if mfcc_value < mfcc_range[0]:
                distance = (mfcc_range[0] - mfcc_value) / mfcc_range[0]
            else:
                distance = (mfcc_value - mfcc_range[1]) / (1 - mfcc_range[1])
            scores['mfcc'] = max(0, 1 - distance)
    
    # Check Zero Crossing Rate
    if 'zero_crossing_rate' in audio_features and 'zero_crossing_rate' in profile:
        zcr_range = profile['zero_crossing_rate']
        zcr_value = audio_features['zero_crossing_rate']
        if zcr_range[0] <= zcr_value <= zcr_range[1]:
            scores['zcr'] = 1.0
        else:
            if zcr_value < zcr_range[0]:
                distance = (zcr_range[0] - zcr_value) / zcr_range[0]
            else:
                distance = (zcr_value - zcr_range[1]) / (1 - zcr_range[1])
            scores['zcr'] = max(0, 1 - distance)
    
    # Check Spectral Centroid
    if 'spectral_centroid' in audio_features and 'spectral_centroid' in profile:
        sc_range = profile['spectral_centroid']
        sc_value = audio_features['spectral_centroid']
        if sc_range[0] <= sc_value <= sc_range[1]:
            scores['spectral_centroid'] = 1.0
        else:
            if sc_value < sc_range[0]:
                distance = (sc_range[0] - sc_value) / sc_range[0]
            else:
                distance = (sc_value - sc_range[1]) / (1 - sc_range[1])
            scores['spectral_centroid'] = max(0, 1 - distance)
    
    # Check Pitch Variation
    if 'pitch_variation' in audio_features and 'pitch_variation' in profile:
        pitch_range = profile['pitch_variation']
        pitch_value = audio_features['pitch_variation']
        if pitch_range[0] <= pitch_value <= pitch_range[1]:
            scores['pitch_variation'] = 1.0
        else:
            if pitch_value < pitch_range[0]:
                distance = (pitch_range[0] - pitch_value) / pitch_range[0]
            else:
                distance = (pitch_value - pitch_range[1]) / (1 - pitch_range[1])
            scores['pitch_variation'] = max(0, 1 - distance)
    
    # Check Tempo
    if 'tempo' in audio_features and 'tempo' in profile:
        tempo_range = profile['tempo']
        tempo_value = audio_features['tempo']
        if tempo_range[0] <= tempo_value <= tempo_range[1]:
            scores['tempo'] = 1.0
        else:
            if tempo_value < tempo_range[0]:
                distance = (tempo_range[0] - tempo_value) / tempo_range[0]
            else:
                distance = (tempo_value - tempo_range[1]) / (1 - tempo_range[1])
            scores['tempo'] = max(0, 1 - distance)
    
    # Calculate average confidence
    if scores:
        avg_confidence = np.mean(list(scores.values()))
    else:
        avg_confidence = 0.5
    
    return avg_confidence, scores