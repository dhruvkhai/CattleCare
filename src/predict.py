import os
import json
import collections
import numpy as np
import pandas as pd
import xgboost as xgb
try:
    from src.features import FEATURE_COLS, CLASS_NAMES, add_engineered_features
except ImportError:
    from features import FEATURE_COLS, CLASS_NAMES, add_engineered_features

class RollingCowBuffer:
    """
    Maintains a rolling FIFO buffer of raw sensor samples (x, y, z) per cow.
    Allows real-time computation of rolling standard deviations, means, min/max, pitch, and roll
    without requiring the ESP32 to compute complex mathematical transforms.
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffers = collections.defaultdict(lambda: collections.deque(maxlen=self.window_size))
        self.last_env = collections.defaultdict(lambda: {"temp": 25.0, "humidity": 60.0, "timestamp": 0})

    def add_sample(self, cow_id, x, y, z, temp=None, humidity=None, timestamp=None):
        self.buffers[cow_id].append({"x": float(x), "y": float(y), "z": float(z)})
        if temp is not None and humidity is not None:
            self.last_env[cow_id] = {
                "temp": float(temp),
                "humidity": float(humidity),
                "timestamp": float(timestamp) if timestamp is not None else 0.0
            }

    def get_window(self, cow_id):
        buf = list(self.buffers[cow_id])
        env = self.last_env[cow_id]
        return buf, env

class CattleCarePredictor:
    def __init__(self, model_path="models/checkpoints/xgboost_cattle_model.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.feature_cols = FEATURE_COLS
        self.class_names = CLASS_NAMES
        self.cow_buffers = RollingCowBuffer(window_size=10) # 1 second at 10 Hz

    def compute_features_from_raw_window(self, readings, temp, humidity):
        """
        Dynamically computes all 32 statistical, kinematic, and environmental features
        from raw MPU6050 (x, y, z) samples and DHT11 (temperature, humidity).
        """
        if not readings:
            readings = [{"x": 0.0, "y": 0.0, "z": 9.8}]
            
        xs = np.array([r['x'] for r in readings], dtype=np.float32)
        ys = np.array([r['y'] for r in readings], dtype=np.float32)
        zs = np.array([r['z'] for r in readings], dtype=np.float32)
        
        # Euclidean acceleration norm
        norms = np.sqrt(xs**2 + ys**2 + zs**2)
        safe_norms = np.where(norms > 1e-4, norms, 1.0)
        
        # Exact CattleCare collar biomechanical posture angles (degrees)
        # roll: lateral collar inclination angle
        rolls = np.arcsin(np.clip(xs / safe_norms, -1.0, 1.0)) * (180.0 / np.pi)
        # pitch: longitudinal head/neck elevation with 30-degree collar mounting calibration offset
        pitches = (np.arctan2(-ys, -zs) * (180.0 / np.pi)) - 30.0
        
        # Barn Temperature-Humidity Index (THI)
        thi = (1.8 * temp + 32.0) - (0.55 - 0.0055 * humidity) * (1.8 * temp - 26.0)
        
        # Standard deviation (0 if only 1 sample)
        x_std = float(np.std(xs)) if len(xs) > 1 else 0.05
        y_std = float(np.std(ys)) if len(ys) > 1 else 0.05
        z_std = float(np.std(zs)) if len(zs) > 1 else 0.05
        norm_std = float(np.std(norms)) if len(norms) > 1 else 0.05
        pitch_std = float(np.std(pitches)) if len(pitches) > 1 else 0.5
        roll_std = float(np.std(rolls)) if len(rolls) > 1 else 0.5
        
        feats = {
            'accel_x_mean': float(np.mean(xs)),
            'accel_x_std': x_std,
            'accel_x_min': float(np.min(xs)),
            'accel_x_max': float(np.max(xs)),
            'accel_x_range': float(np.max(xs) - np.min(xs)),
            
            'accel_y_mean': float(np.mean(ys)),
            'accel_y_std': y_std,
            'accel_y_min': float(np.min(ys)),
            'accel_y_max': float(np.max(ys)),
            'accel_y_range': float(np.max(ys) - np.min(ys)),
            
            'accel_z_mean': float(np.mean(zs)),
            'accel_z_std': z_std,
            'accel_z_min': float(np.min(zs)),
            'accel_z_max': float(np.max(zs)),
            'accel_z_range': float(np.max(zs) - np.min(zs)),
            
            'accel_norm_mean': float(np.mean(norms)),
            'accel_norm_std': norm_std,
            'accel_norm_max': float(np.max(norms)),
            
            'pitch_mean': float(np.mean(pitches)),
            'pitch_std': pitch_std,
            'pitch_min': float(np.min(pitches)),
            'pitch_max': float(np.max(pitches)),
            'pitch_range': float(np.max(pitches) - np.min(pitches)),
            
            'roll_mean': float(np.mean(rolls)),
            'roll_std': roll_std,
            
            'collar_temp_C': float(temp),
            'barn_temp_C': float(temp),
            'barn_humidity_per': float(humidity),
            'THI': float(thi),
            'temp_diff': 0.0,
            
            'dynamic_body_accel': float(x_std + y_std + z_std),
            'chew_intensity': float(pitch_std * y_std)
        }
        return feats

    def predict_raw_esp32(self, x, y, z, temperature, humidity, timestamp=None, cow_id="COW_ESP32"):
        """
        Primary dynamic entrypoint for ESP32.
        Ingests only the raw hardware readings (x, y, z, temperature, humidity).
        Maintains real-time buffer per cow and calculates all features dynamically.
        """
        self.cow_buffers.add_sample(cow_id, x, y, z, temperature, humidity, timestamp)
        readings, env = self.cow_buffers.get_window(cow_id)
        
        dynamic_features = self.compute_features_from_raw_window(readings, env['temp'], env['humidity'])
        
        # Build feature vector matching model order
        df = pd.DataFrame([dynamic_features])
        X = df[self.feature_cols].values
        probs = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        thi = dynamic_features['THI']
        heat_stress_status = "Normal"
        if thi >= 79:
            heat_stress_status = "Severe Heat Stress"
        elif thi >= 72:
            heat_stress_status = "Mild Heat Stress"
            
        return {
            "cow_id": cow_id,
            "timestamp": timestamp,
            "predicted_behavior": pred_label,
            "confidence": round(confidence * 100, 2),
            "buffer_depth": len(readings),
            "telemetry": {
                "pitch_angle": round(dynamic_features['pitch_mean'], 1),
                "roll_angle": round(dynamic_features['roll_mean'], 1),
                "movement_intensity": round(dynamic_features['accel_norm_mean'], 2),
                "THI": round(thi, 1),
                "heat_stress_level": heat_stress_status
            },
            "class_probabilities": {name: round(float(p) * 100, 2) for name, p in zip(self.class_names, probs)}
        }

    def predict_sample(self, raw_features_dict):
        """
        Fallback compatibility method for pre-computed feature dictionaries.
        """
        df = pd.DataFrame([raw_features_dict])
        df = add_engineered_features(df)
        X = df[self.feature_cols].values
        probs = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
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
