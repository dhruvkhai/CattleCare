# CattleCare: Precision Livestock AI Monitoring System

CattleCare is an AI-powered livestock health and behavior monitoring platform designed for an **IoT neckband collar** equipped with:
- **MPU6050**: 6-axis Inertial Measurement Unit (3-axis accelerometer and gyroscope).
- **DHT11**: Microclimate ambient temperature and relative humidity sensor.
- **ESP32**: Edge microcontroller for signal acquisition, processing, and wireless telemetry.
- **RFID**: Individual animal identification (`cow_id`).

---

## 1. Quick Start: How to Run the Applications

### Option A: Interactive Web Monitoring Dashboard (Recommended)
Launch the real-time web monitoring interface:
```bash
python app.py
```
Then open your web browser at:
```
http://127.0.0.1:8000
```
**Features of the Web Dashboard:**
- **Live Collar Telemetry**: Streams real-time sensor packets from unseen test cows (`C09`, `C10`).
- **Real-Time KPIs**: Current behavior prediction, confidence percentage, neck pitch angle, movement intensity, and barn THI heat stress level.
- **Daily Behavioral Time-Budget**: Interactive donut chart displaying the breakdown of daily activity (Feeding, Resting, Standing, Walking, Drinking, Licking).
- **Cow Selector**: Switch between different animals to inspect individual behavioral profiles.

---

### Option B: Command-Line Live Stream Simulation (CLI)
Simulate real-time collar sensor streaming directly in the terminal:
```bash
# Stream 10 live seconds from unseen test cow C09:
python run_demo.py --cow C09 --samples 10 --delay 0.5

# Test cow C10 with fast streaming:
python run_demo.py --cow C10 --samples 20 --delay 0.2
```

---

### Option C: Python Inference API (Embed into Custom Code)
To make predictions on real-time sensor packets in your own scripts:

```python
from src.predict import CattleCarePredictor

predictor = CattleCarePredictor()

# Example: 1-second aggregated collar sensor readings
sensor_packet = {
    'accel_x_mean': -4.2, 'accel_x_std': 0.5, 'accel_x_min': -5.0, 'accel_x_max': -3.5,
    'accel_y_mean': 1.2,  'accel_y_std': 0.8, 'accel_y_min': 0.5,  'accel_y_max': 2.0,
    'accel_z_mean': 8.5,  'accel_z_std': 0.6, 'accel_z_min': 7.5,  'accel_z_max': 9.5,
    'accel_norm_mean': 9.8, 'accel_norm_std': 0.7, 'accel_norm_max': 10.5,
    'pitch_mean': -62.0,  'pitch_std': 5.2,  'pitch_min': -70.0, 'pitch_max': -55.0,
    'roll_mean': -5.0,   'roll_std': 3.0,
    'collar_temp_C': 27.5, 'barn_temp_C': 24.0, 'barn_humidity_per': 65.0, 'THI': 73.5
}

result = predictor.predict_sample(sensor_packet)
print("Prediction:", result['predicted_behavior'])
print("Confidence:", result['confidence'], "%")
print("Heat Stress Status:", result['environmental_status']['heat_stress_level'])
```

---

## 2. Re-Running Model Training & Evaluation Pipelines

All training pipelines utilize **Group-Based Cross-Validation (`GroupKFold` across `cow_id`)** to prevent data leakage:

```bash
# 1. Train GPU-Accelerated XGBoost model:
python src/train_xgboost.py

# 2. Train Deep Learning Neural Network (CattleNet on CUDA):
python src/train_deep_model.py

# 3. Train 15-day longitudinal posture monitoring model (Lying vs Standing):
python src/train_posture_model.py

# 4. Run comparative evaluation, ensembling, and generate report plots:
python src/evaluate_ensemble.py
```

---

## 3. Directory Layout

```
CattleCare/
├── app.py                                   <-- Interactive Web Dashboard (FastAPI)
├── run_demo.py                              <-- Interactive CLI Streaming Simulator
├── README.md                                <-- System documentation
│
├── src/                                     <-- Core Source Code
│   ├── features.py                          <-- Feature engineering & hardware mappings
│   ├── predict.py                           <-- Real-time inference engine
│   ├── train_xgboost.py                     <-- XGBoost training pipeline
│   ├── train_deep_model.py                  <-- CattleNet PyTorch deep learning pipeline
│   ├── train_posture_model.py               <-- 15-day continuous posture model
│   └── evaluate_ensemble.py                 <-- Ensembling & figure generation suite
│
├── models/checkpoints/                      <-- Saved Model Artifacts
│   ├── xgboost_cattle_model.json            <-- Trained XGBoost model (~1.8 MB)
│   ├── cattlenet_deep_model.pt              <-- Trained PyTorch deep model (~917 KB)
│   ├── posture_monitoring_model.json        <-- Trained posture model (~1.4 MB)
│   └── ensemble_metrics.json                <-- Comprehensive test evaluation metrics
│
├── data/
│   ├── analysis/                            <-- Dataset inventory, mappings & exclusion logs
│   ├── processed/
│   │   ├── hardware_compatible/             <-- Clean datasets ready for training
│   │   └── excluded_research_data/          <-- Archived unsupported sensors
│   └── reports/
│       ├── model_training_and_evaluation_report.md
│       ├── dataset_cleaning_report.md
│       └── figures/                         <-- Confusion matrix & feature importance plots
│
└── sensor_data/                             <-- Original Kaggle dataset (read-only reference)
```
