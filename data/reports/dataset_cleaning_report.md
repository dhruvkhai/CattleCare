# CattleCare: Final Dataset Cleaning, Validation & Provenance Report

**Date:** 2026-11-18  
**Author:** Senior Data Engineer & Machine Learning Engineer  
**Project:** CattleCare Precision Livestock Monitoring System  
**Hardware Target:** CattleCare IoT Neckband (DHT11, MPU6050, ESP32, RFID)  
**Input Source:** `sensor_data/` (MmCows Multimodal Dairy Dataset, NEIS Lab / NeurIPS)  
**Processed Output:** `data/processed/hardware_compatible/`  

---

## 1. Executive Summary

This report documents the completion of the dataset cleaning, hardware compatibility selection, transformation, and rigorous validation for the CattleCare machine learning pipeline.

In accordance with strict data engineering standards and project safety rules:
1. **Raw Preservation:** All **1,171 original Kaggle files** (16.88 GB, 438,456,823 raw records) in `sensor_data/` remain **completely unaltered and unmodified**.
2. **Hardware Alignment:** Every attribute in the processed datasets is directly measurable or mathematically derivable from our actual **DHT11** (ambient temperature/humidity), **MPU6050** (3-axis accelerometer/gyroscope), **ESP32** (microcontroller timer/processing), and **RFID** (`cow_id`).
3. **No Synthetic Data:** Zero synthetic sensor readings were generated. All values reflect verified physical recordings.
4. **Target Modularity:** In strict compliance with ML objectives, the **7-class ethological behavior classification model** is isolated as a modular behavioral activity model. It is **not** characterized as a disease prediction model. Posture monitoring (lying vs standing) is structured as a separate continuous monitoring pipeline across all 15 days.

---

## 2. Record Counts: Before vs. After Processing

| Pipeline / Dataset Component | Raw Observations Inspected | Clean Output Records | Dimensionality (Features) | Output File Name & Format |
| :--- | :---: | :---: | :---: | :--- |
| **High-Frequency Behavior Dataset (10 Hz)** | 8,640,000 (10 cows $\times$ 864k rows) | **6,999,723** | 14 columns | `behavior_training_10hz.csv` (625.89 MB)<br>`behavior_training_10hz.csv.gz` (112.11 MB) |
| **Edge Windowed Behavior Dataset (1-Second)** | 864,000 (10 cows $\times$ 86.4k seconds) | **704,665** | 30 columns | `behavior_training_windowed_1s.csv` (199.44 MB) |
| **Continuous Posture Monitoring Dataset (1-Minute)** | 197,920 (10 cows $\times$ 15 days) | **197,920** | 8 columns | `posture_monitoring_15days_1min.csv` (12.17 MB) |
| **Total Clean Records Generated** | **9,701,920** | **7,902,308** | Fully Verified | Total Output Directory Size: ~950 MB |

### Why Did 10 Hz Records Reduce from 8,640,000 to 6,999,723?
The reduction of 1,640,277 records (-18.98%) across the 10 animals on July 25, 2023 is accounted for by two strictly necessary quality filters:
1. **Exclusion of Class 0 (Unknown / Camera Occluded)**: 1,519,840 records (17.59% of raw frames) represented video occlusions (cows behind feed bunks, pillars, or during rapid unlabeled transition states). Per user approval of Option A, these were excluded so the supervised classifier trains exclusively on validated ethological ground truth.
2. **Exclusion of Sensor Missing Values (NaNs)**: 120,437 records (1.39%) contained missing accelerometer packets from intermittent wireless transmission drops. These were removed to guarantee complete numerical vectors.

---

## 3. Generated Dataset Schemas & Feature Definitions

### 3.1 Dataset 1: `behavior_training_10hz.csv` (and `.csv.gz`)
High-resolution 10 Hz synchronized sensor observations for July 25, 2023 across cows C01–C10:

| Column Name | Data Type | Hardware Origin | Measurement Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `float64` | ESP32 System Time | Unix Epoch (seconds) | High-precision timestamp at 0.1s step (e.g., 1690272000.1) |
| `cow_id` | `string` | RFID Identifier | Categorical (`C01`–`C10`)| Unique animal identification |
| `accel_x_mps2` | `float64` | MPU6050 Accelerometer | $\text{m/s}^2$ | Forward / backward neck acceleration |
| `accel_y_mps2` | `float64` | MPU6050 Accelerometer | $\text{m/s}^2$ | Lateral side-to-side neck acceleration |
| `accel_z_mps2` | `float64` | MPU6050 Accelerometer | $\text{m/s}^2$ | Vertical head nodding acceleration |
| `pitch` | `float64` | MPU6050 Derived Angle | Degrees ($^\circ$) | Head tilt angle: $\approx -61.5^\circ$ (feeding) vs $\approx -9.0^\circ$ (lying) |
| `roll` | `float64` | MPU6050 Derived Angle | Degrees ($^\circ$) | Lateral neck tilt angle (grooming, licking) |
| `accel_norm` | `float64` | MPU6050 Derived Norm | $\text{m/s}^2$ | Euclidean acceleration magnitude $\sqrt{a_x^2 + a_y^2 + a_z^2}$ |
| `collar_temp_C`| `float64` | Collar Enclosure Temp | $^\circ\text{C}$ | Temperature at collar casing (median-filtered) |
| `barn_temp_C` | `float64` | DHT11 Environmental | $^\circ\text{C}$ | Ambient temperature in barn microclimate |
| `barn_humidity_per`| `float64`| DHT11 Environmental | Percentage (%) | Relative humidity in barn microclimate |
| `THI` | `float64` | ESP32 Derived Index | Dimensionless | Temperature-Humidity Index for cattle heat stress |
| `behavior` | `int64` | Video Ground Truth | Discrete (1–7) | Supervised target label |
| `behavior_name`| `string` | Label Annotation | String Label | Human-readable behavior class |

### 3.2 Dataset 2: `behavior_training_windowed_1s.csv`
1-second aggregated feature vectors (10 samples per window, minimum 8 valid samples required) optimized for edge inference on ESP32:

- **Statistical Accelerometry Features (12 features):**
  - `accel_x_mean`, `accel_x_std`, `accel_x_min`, `accel_x_max`
  - `accel_y_mean`, `accel_y_std`, `accel_y_min`, `accel_y_max`
  - `accel_z_mean`, `accel_z_std`, `accel_z_min`, `accel_z_max`
- **Dynamic Movement Intensity (3 features):**
  - `accel_norm_mean`, `accel_norm_std`, `accel_norm_max`
- **Posture & Orientation (6 features):**
  - `pitch_mean`, `pitch_std`, `pitch_min`, `pitch_max`
  - `roll_mean`, `roll_std`
- **Microclimate & Thermal Features (4 features):**
  - `collar_temp_C`, `barn_temp_C`, `barn_humidity_per`, `THI`
- **Metadata & Targets (5 columns):**
  - `timestamp`, `cow_id`, `sample_count`, `behavior`, `behavior_name`

### 3.3 Dataset 3: `posture_monitoring_15days_1min.csv`
Continuous 15-day longitudinal monitoring dataset at 1-minute epoch resolution across all 10 animals (197,920 records):

| Column Name | Data Type | Hardware Channel | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `int64` | ESP32 Epoch Minute | Integer Unix epoch at 60s step |
| `datetime` | `string` | Chronological | Formatted string (e.g., `21-07-2023 12:30`) |
| `cow_id` | `string` | RFID Identifier | Animal identifier (`C01`–`C10`) |
| `collar_temp_C`| `float64`| Collar Sensor | Average collar casing temperature over epoch |
| `barn_temp_C` | `float64`| DHT11 Ambient | Ambient barn temperature ($14.4^\circ\text{C}$ to $31.7^\circ\text{C}$) |
| `barn_humidity_per`| `float64`| DHT11 Ambient | Relative humidity ($46.3\%$ to $97.3\%$) |
| `THI` | `float64`| ESP32 Derived | Temperature-Humidity Index ($58.0$ to $83.1$) |
| `lying` | `int64` | Ankle Ground Truth | Binary posture target: **1 = Lying / Resting**, **0 = Standing / Active** |

---

## 4. Final Class Distributions & Target Characteristics

### 4.1 7-Class Behavior Target Distribution (1-Second Dataset: 704,665 Samples)

| Code | Behavior Class | Sample Count | Percentage | Primary Distinguishing Features |
| :---: | :--- | :---: | :---: | :--- |
| **7** | **Lying** | 359,916 | 51.08% | Near-horizontal pitch ($\approx -9.0^\circ$), low `accel_norm_std` |
| **2** | **Standing** | 188,981 | 26.82% | Moderate upright pitch ($\approx -35.0^\circ$), low dynamic variance |
| **4** | **Feeding (Head Down)** | 79,937 | 11.34% | Steep downward pitch ($\approx -50^\circ$ to $-72^\circ$), nodding variance |
| **3** | **Feeding (Head Up)** | 51,132 | 7.26% | Elevated head pitch ($\approx -31.9^\circ$), rhythmic chewing cyclic oscillations |
| **1** | **Walking** | 10,525 | 1.49% | Highest dynamic acceleration norm and jerk variance |
| **6** | **Drinking** | 8,322 | 1.18% | Static lowered head pitch ($\approx -45^\circ$) near water trough |
| **5** | **Licking** | 5,852 | 0.83% | Pronounced roll angle standard deviation |

### 4.2 Posture Target Distribution (15-Day Dataset: 197,920 Samples)
- **Class 1 (Lying / Resting)**: 108,037 samples (**54.59%**)
- **Class 0 (Standing / Active)**: 89,883 samples (**45.41%**)
- **Balance Assessment**: Highly balanced binary target, ideal for evaluating rest duration, lying bouts, and restlessness indicators.

---

## 5. Comprehensive Quality & Integrity Checks

All generated files were programmatically verified using automated validation suites:
- **Null Value Check**: **0 missing values** across all rows and columns in all three processed tables.
- **Timestamp Uniqueness Check**: **0 duplicate timestamps** per cow. Chronological monotonicity strictly verified.
- **Hardware Isolation Check**: Confirmed that all unsupported channels (`mag_x/y/z`, `yaw`, `cbt`, `pressure_Pa`, `elevation_m`, `coord_x/y/z_cm`, `distance_m`, `milk_weight_kg`, `DIM`, `loc_x/y/z_cm`) have been **100% eliminated** from the hardware-compatible datasets.
- **Raw File Integrity Check**: Confirmed that all 1,171 files in `sensor_data/` remain byte-identical to their original uploaded state.

---

## 6. Provenance & Excluded Research Data Catalog

All unsupported modalities have been cataloged in `data/processed/excluded_research_data/research_archive_manifest.csv` and [exclusion_log.csv](file:///c:/Users/dhruv/CattleCare/data/analysis/exclusion_log.csv):

1. `cbt` (10 files, 197,920 rows): Internal core body temperature implants.
2. `pressure` (180 files, 142,270,067 rows): Barometric pressure and elevation.
3. `uwb` & `uwb_distance` (360 files, 8,502,831 rows): UWB spatial coordinates and anchor ranging.
4. `ankle_accel` (10 files, 197,920 rows): 3-axis leg acceleration.
5. `milk` (16 files, 240 rows): Milking parlor production data.
6. `visual_location` (16 files, 80,640 rows): Computer vision 3D localization.
7. `weather` (17 files): Regional weather station spreadsheets.
8. `stationary_tags_T13_T14` (60 files, 47,423,356 rows): Barn ceiling reference transceivers.
9. `unmatched_visual_labels_C11_C16` (6 files, 518,400 rows): Visual annotations for cows without collar sensors.

---

## 7. Remaining Data Limitations & Engineering Recommendations

1. **Behavior Ground Truth Duration**:
   Ground-truth ethological 7-class video annotations are available exclusively for **July 25, 2023** (24 hours). This benchmark day provides 704,665 1-second samples across 10 animals, which is substantial for training modern classification architectures (Random Forest, XGBoost, 1D-CNN, or Temporal Convolutional Networks). For the remaining 14 days, continuous ground truth is available for **posture (lying vs standing)**.
2. **Absence of Acute Disease Labels**:
   As verified, no acute clinical disease diagnoses occurred during the 14-day study period. The behavior model must be trained as an **ethological activity recognition engine**. Abnormality detection (e.g., detecting decreased feeding, excessive lying, or restlessness indicative of lameness or illness) should be designed as a subsequent heuristic or anomaly-detection layer operating on predicted daily behavioral time-budgets.
3. **Collar Glitch Filtering**:
   Intermittent `-1.0` glitch readings in raw neck casing temperature were successfully filtered using forward-fill and median filtering.
