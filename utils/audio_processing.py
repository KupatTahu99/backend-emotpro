import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import io


def process_audio_csv(csv_file: io.BytesIO) -> Tuple[Optional[Dict], Dict]:
    try:
        df = pd.read_csv(csv_file)
        
        if df.empty:
            return None, {"samples": 0}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return None, {"samples": 0}
        
        features = {}
        
        if 'amplitude' in df.columns:
            amplitude = df['amplitude'].values
            features['amplitude_mean'] = float(np.mean(amplitude))
            features['amplitude_std'] = float(np.std(amplitude))
            features['amplitude_max'] = float(np.max(amplitude))
            features['amplitude_energy'] = float(np.sum(amplitude ** 2))
        
        mfcc_cols = [col for col in numeric_cols if 'mfcc' in col.lower()]
        if mfcc_cols:
            mfcc_data = df[mfcc_cols].values
            features['mfcc_mean'] = float(np.mean(mfcc_data))
            features['mfcc_std'] = float(np.std(mfcc_data))
            features['mfcc_coefficients'] = int(len(mfcc_cols))
        
        if 'energy' in df.columns:
            energy = df['energy'].values
            features['energy_mean'] = float(np.mean(energy))
            features['energy_std'] = float(np.std(energy))
        
        if 'zero_crossing_rate' in df.columns:
            zcr = df['zero_crossing_rate'].values
            features['zero_crossing_rate'] = float(np.mean(zcr))
        
        if 'spectral_centroid' in df.columns:
            sc = df['spectral_centroid'].values
            features['spectral_centroid'] = float(np.mean(sc))
        
        if not features:
            for col in numeric_cols[:5]:
                features[col] = float(df[col].mean())
        
        metadata = {
            "samples": len(df),
            "total_columns": len(df.columns),
            "numeric_features": len(numeric_cols)
        }
        
        return features, metadata
        
    except pd.errors.ParserError:
        return None, {"samples": 0, "error": "Invalid CSV format"}
    except Exception as e:
        return None, {"samples": 0, "error": str(e)}