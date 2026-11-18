import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

sys.path.append('src')
from predict import CattleCarePredictor

def run_cli():
    parser = argparse.ArgumentParser(description="CattleCare Live Inference & Demonstration CLI")
    parser.add_argument("--cow", type=str, default="C09", help="Cow ID to simulate (e.g. C09, C10)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to simulate")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between streaming samples (seconds)")
    args = parser.parse_args()

    print("="*75)
    print("      CATTLECARE - PRECISION LIVESTOCK EDGE AI INFERENCE ENGINE      ")
    print("      Hardware: ESP32 + MPU6050 (6-DOF IMU) + DHT11 (Temp/Humidity)   ")
    print("="*75)
    
    predictor = CattleCarePredictor()
    print("Loaded trained model: models/checkpoints/xgboost_cattle_model.json")
    
    # Load test data
    data_path = r"data\processed\hardware_compatible\behavior_training_windowed_1s.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        return

    print(f"Reading sensor stream for Cow: {args.cow}...")
    df = pd.read_csv(data_path)
    cow_df = df[df['cow_id'] == args.cow].copy()
    
    if len(cow_df) == 0:
        print(f"Cow {args.cow} not found! Available cows: {df['cow_id'].unique().tolist()}")
        return

    sample_df = cow_df.sample(min(args.samples, len(cow_df)), random_state=42)
    
    print("\n" + "-"*85)
    print(f"{'Time':<12} | {'Neck Pitch':<11} | {'Movement':<9} | {'THI':<6} | {'Ground Truth':<20} | {'Predicted Behavior':<20} | {'Conf':<6}")
    print("-"*85)
    
    correct = 0
    total = 0
    
    for idx, row in sample_df.iterrows():
        sample_dict = row.to_dict()
        res = predictor.predict_sample(sample_dict)
        
        gt = row['behavior_name']
        pred = res['predicted_behavior']
        conf = res['confidence']
        pitch = f"{row['pitch_mean']:.1f}°"
        accel = f"{row['accel_norm_mean']:.2f} m/s²"
        thi = f"{row['THI']:.1f}"
        ts = str(row['timestamp'])[-5:]
        
        match = "✓" if gt == pred else "✗"
        if gt == pred:
            correct += 1
        total += 1
        
        print(f"{ts:<12} | {pitch:<11} | {accel:<9} | {thi:<6} | {gt:<20} | {pred:<20} | {conf:.1f}% {match}")
        if args.delay > 0:
            time.sleep(args.delay)
            
    print("-"*85)
    print(f"Batch Simulation Finished: {correct}/{total} correct ({correct/total*100:.1f}% accuracy on unseen cow {args.cow})")
    print("="*75)

if __name__ == "__main__":
    run_cli()
