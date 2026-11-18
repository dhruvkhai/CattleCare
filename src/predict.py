import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from features import FEATURE_COLS, CLASS_NAMES, add_engineered_features

class CattleCarePredictor:
    def __init__(self, model_path="models/checkpoints/xgboost_cattle_model.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.feature_cols = FEATURE_COLS
        self.class_names = CLASS_NAMES

    def predict_sample(self, raw_features_dict):
        """
        Takes raw 1-second sensor aggregations and outputs prediction with probabilities.
        """
        df = pd.DataFrame([raw_features_dict])
        df = add_engineered_features(df)
        
        # Verify all required features exist
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
            
        X = df[self.feature_cols].values
        probs = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Heat stress warning based on THI
        thi = float(df['THI'].iloc[0]) if 'THI' in df.columns else 70.0
        heat_stress_status = "Normal"
        if thi >= 79:
            heat_stress_status = "Severe Heat Stress"
        elif thi >= 72:
            heat_stress_status = "Mild Heat Stress"
            
        return {
            "predicted_behavior": pred_label,
            "confidence": round(confidence * 100, 2),
            "class_probabilities": {name: round(float(p) * 100, 2) for name, p in zip(self.class_names, probs)},
            "environmental_status": {
                "THI": round(thi, 1),
                "heat_stress_level": heat_stress_status
            }
        }

    def predict_batch(self, df):
        """
        Batch prediction for a dataframe of 1-second records.
        """
        df_feats = add_engineered_features(df)
        X = df_feats[self.feature_cols].values
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        
        df_out = df.copy()
        df_out['predicted_behavior'] = [self.class_names[i] for i in preds]
        df_out['confidence'] = np.max(probs, axis=1) * 100.0
        return df_out
