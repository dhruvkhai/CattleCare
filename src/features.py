import os
import pandas as pd
import numpy as np

FEATURE_COLS = [
    'accel_x_mean', 'accel_x_std', 'accel_x_min', 'accel_x_max', 'accel_x_range',
    'accel_y_mean', 'accel_y_std', 'accel_y_min', 'accel_y_max', 'accel_y_range',
    'accel_z_mean', 'accel_z_std', 'accel_z_min', 'accel_z_max', 'accel_z_range',
    'accel_norm_mean', 'accel_norm_std', 'accel_norm_max',
    'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',
    'roll_mean', 'roll_std',
    'collar_temp_C', 'barn_temp_C', 'barn_humidity_per', 'THI', 'temp_diff',
    'dynamic_body_accel', 'chew_intensity'
]

LABEL_MAP = {
    1: 0, # Walking
    2: 1, # Standing
    3: 2, # Feeding (Head Up)
    4: 3, # Feeding (Head Down)
    5: 4, # Licking
    6: 5, # Drinking
    7: 6  # Lying
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

CLASS_NAMES = [
    "Walking",
    "Standing",
    "Feeding (Head Up)",
    "Feeding (Head Down)",
    "Licking",
    "Drinking",
    "Lying"
]

def add_engineered_features(df):
    """
    Computes domain-specific kinematic and thermal features compatible with CattleCare hardware.
    """
    df = df.copy()
    df['accel_x_range'] = df['accel_x_max'] - df['accel_x_min']
    df['accel_y_range'] = df['accel_y_max'] - df['accel_y_min']
    df['accel_z_range'] = df['accel_z_max'] - df['accel_z_min']
    df['pitch_range'] = df['pitch_max'] - df['pitch_min']
    df['temp_diff'] = df['collar_temp_C'] - df['barn_temp_C']
    df['dynamic_body_accel'] = df['accel_x_std'] + df['accel_y_std'] + df['accel_z_std']
    df['chew_intensity'] = df['pitch_std'] * df['accel_y_std']
    return df

def prepare_data(data_path=None):
    if data_path is None:
        candidates = [
            r"data\processed\hardware_compatible\behavior_training_windowed_1s.csv",
            r"data\processed\hardware_compatible\behavior_training_windowed_1s.csv.gz",
            r"data\processed\hardware_compatible\sample_behavior_data.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError("Could not find hardware_compatible behavior training dataset.")
        
    df = pd.read_csv(data_path)
    df = add_engineered_features(df)
    df['target'] = df['behavior'].map(LABEL_MAP)
    return df
