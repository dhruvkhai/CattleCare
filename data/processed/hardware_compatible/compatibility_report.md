# CattleCare Hardware Compatibility & Dataset Inspection Report

**Date:** 2026-11-18  
**Author:** Senior Data Engineer & Machine Learning Engineer  
**Project:** CattleCare Precision Livestock Monitoring System  
**Dataset Analyzed:** `sensor_data/` (MmCows: Dairy Cattle Multimodal Dataset, NEIS Lab / NeurIPS)  

---

## 1. Executive Summary

A comprehensive, recursive inspection was conducted across the entire Kaggle dataset uploaded to `sensor_data/`. The dataset comprises **1,171 individual files**, totaling **16.88 GB** and **438,456,823 sensor observations** spanning a 14-day study period from **July 21, 2023 to August 04, 2023** at the University of Wisconsin-Madison Arlington Agricultural Research Station.

The CattleCare hardware architecture is an **IoT neckband collar** consisting of:
- **DHT11:** Ambient temperature and relative humidity.
- **MPU6050:** 6-axis IMU (3-axis accelerometer and 3-axis gyroscope) measuring neck dynamics, posture angles, and movement intensity.
- **ESP32:** Edge micro-controller for acquisition, signal processing, and Wi-Fi transmission.
- **RFID:** Individual animal identification (`cow_id`).

Through rigorous schema, temporal, and biological analysis, every file and attribute has been categorized according to hardware compatibility:
- **Category A (Directly Compatible):** 157 files (Neck 3-axis accelerometer, barn ambient temperature, relative humidity).
- **Category B (Derivable / Conditionally Compatible):** 160 files (Head pitch, roll, acceleration norm, Temperature-Humidity Index (THI), neck collar temperature).
- **Category C (Incompatible):** 609 files (Core Body Temperature implants, barometric pressure/elevation, UWB 3D positioning, UWB anchor ranging, leg/ankle acceleration, milking parlor yields, overhead computer vision).
- **Category D (Target & Supporting Data):** 182 files (Ground-truth ethological behavior labels, ankle lying posture labels, historical veterinary health records).
- **Category E (Unnecessary / Redundant):** 63 files (Stationary barn calibration tags T13/T14, reference documentation images, non-tabular weather summaries).

---

## 2. Dataset Architecture & Folder Breakdown

The raw directory `sensor_data/` is structured into three primary branches: `behavior_labels/`, `main_data/`, and `sub_data/`.

| Modality / Folder | File Count | Total Records | Total Size | Sampling Frequency | Available Attributes | Category | Recommended Action |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| `behavior_labels/individual` | 16 | 1,382,400 | 30.60 MB | 1 Hz (1 sec) | `timestamp`, `datetime`, `behavior` | **D** | **KEEP** (C01-C10) / **INVESTIGATE** (C11-C16) |
| `main_data/immu` | 180 | 142,270,067 | 6.93 GB | 10 Hz (0.1 sec) | `timestamp`, `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2`, `mag_x_uT`, `mag_y_uT`, `mag_z_uT` | **A / C** | **KEEP** (T01-T10, drop `mag_*`) |
| `sub_data/head_direction` | 181 | 142,270,067 | 5.84 GB | 10 Hz (0.1 sec) | `timestamp`, `roll`, `pitch`, `yaw`, `accel_norm`, `relative_angle` (+ 1 jpg) | **B / C** | **KEEP** (T01-T10, drop `yaw`) |
| `main_data/thi` | 7 | 138,486 | 4.48 MB | 1/60 Hz (1 min) | `timestamp`, `temperature_C`, `humidity_per`, `THI` | **A / B** | **KEEP** |
| `sub_data/neck_dev_temp` | 12 | 944,759 | 17.92 MB | ~1/15 Hz (15 sec)| `timestamp`, `temperature_C` | **B** | **KEEP** (T01-T10) |
| `main_data/ankle` | 150 | 197,920 | 6.79 MB | 1/60 Hz (1 min) | `timestamp`, `datetime`, `lying` | **D** | **KEEP** (Ground truth posture) |
| `sub_data/health_records` | 16 | 3,521 | 0.17 MB | Irregular event | `Date`, `Event`, `Specific event`, `Description` | **D** | **KEEP** (Metadata / Traceability) |
| `main_data/cbt` | 10 | 197,920 | 3.72 MB | 1/60 Hz (1 min) | `timestamp`, `temperature_C` | **C** | **EXCLUDE** (Research Archive) |
| `main_data/pressure` | 180 | 142,270,067 | 4.14 GB | 10 Hz (0.1 sec) | `timestamp`, `pressure_Pa`, `elevation_m` | **C** | **EXCLUDE** (Research Archive) |
| `main_data/uwb` | 180 | 944,759 | 28.35 MB | Event / ~15 sec | `timestamp`, `coord_x_cm`, `coord_y_cm`, `coord_z_cm` | **C** | **EXCLUDE** (Research Archive) |
| `sub_data/uwb_distance` | 180 | 7,558,072 | 274.79 MB | Multi-anchor | `timestamp`, `anchor_id`, `distance_m`, `n_samples`, `LOS_probability`, `RSSI_dBm`| **C** | **EXCLUDE** (Research Archive) |
| `sub_data/ankle_accel` | 10 | 197,920 | 5.67 MB | 1/60 Hz (1 min) | `timestamp`, `accel_x`, `accel_y`, `accel_z` | **C** | **EXCLUDE** (Research Archive) |
| `main_data/milk` | 16 | 240 | 0.01 MB | 1/day | `timestamp`, `milk_weight_kg`, `DIM` | **C** | **EXCLUDE** (Research Archive) |
| `sub_data/visual_location` | 16 | 80,640 | 2.43 MB | Video frames | `timestamp`, `loc_x_cm`, `loc_y_cm`, `loc_z_cm` | **C** | **EXCLUDE** (Research Archive) |
| `main_data/weather` | 17 | N/A | 8.64 MB | Irregular/Daily | Excel sheets, maps.png, docx | **C / E** | **EXCLUDE** (Research Archive) |

---

## 3. Hardware Compatibility Assessment

### 3.1 DHT11 Sensor Compatibility
- **Ambient Barn Temperature & Humidity (`main_data/thi`):** Fully compatible. DHT11 directly measures ambient air temperature and relative humidity. Note: In `main_data/thi/average.csv`, the column `temperature_F` is misnamed in the Kaggle release; values are in Celsius (mean 23.19°C) and match sensors S01–S06.
- **Collar Temperature (`sub_data/neck_dev_temp`):** Conditionally compatible. Reflects ambient microclimate and device enclosure heating at the cow's neck. Contains intermittent `-1.0` glitch values that require filtering.
- **Core Body Temperature (`main_data/cbt`):** **INCOMPATIBLE.** CBT measures internal physiological temperature (rectal or reticulorumen bolus, 37.5°C–39.5°C). DHT11 cannot measure internal core body temperature. Must be excluded from the collar ML training feature set.

### 3.2 MPU6050 6-Axis IMU Compatibility
- **3-Axis Acceleration (`main_data/immu`):** Fully compatible. `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2` directly match MPU6050 accelerometer channels.
- **Magnetometer Readings (`mag_x_uT`, `mag_y_uT`, `mag_z_uT`):** **INCOMPATIBLE.** MPU6050 has no magnetic sensor. These three columns must be stripped.
- **Derived Biomechanical Angles (`sub_data/head_direction`):**
  - `pitch` ($\arctan(-a_x / \sqrt{a_y^2 + a_z^2})$): **Directly derivable.** Strongly correlates with feeding (mean pitch: -61.5° when feeding head down vs -9.0° when lying).
  - `roll` ($\arctan(a_y / a_z)$): **Directly derivable.** Measures head tilting and grooming.
  - `accel_norm` ($\sqrt{a_x^2 + a_y^2 + a_z^2}$): **Directly derivable.** Measures overall movement intensity and kinetic activity.
  - `yaw`: **INCOMPATIBLE.** Absolute heading cannot be maintained without a magnetometer. Must be excluded.

### 3.3 ESP32 & RFID
- **Traceability:** RFID uniquely identifies individual cattle (`cow_id` C01 to C10).
- **Signal & Power:** Wi-Fi RSSI and battery voltage are ESP32 operational parameters. The raw Kaggle dataset does not record Wi-Fi RSSI or ESP32 battery level (UWB RSSI in `uwb_distance` reflects Decawave anchor signals and is incompatible).

---

## 4. Cross-Sensor Relationships & Animal Mapping

### 4.1 Biological Subject (`C`) vs Hardware Tag (`T`) Mapping
A key finding of this investigation is the mapping between hardware tags and animals:
- Tags **T01 through T10** correspond directly to cows **C01 through C10** ($T_i \leftrightarrow C_i$).
- Tags **T13 and T14** are stationary reference transceivers mounted on barn roof structures, not attached to animals.
- Cows **C11 through C16** were visually monitored by video cameras and have behavior labels, milk records, and health records, but wore **no neck IMMU sensors** in this dataset.

### 4.2 Temporal Alignment & Sampling Synchronization
- **Epoch Timestamps:** Timestamps across all CSV files are synchronized Unix epoch seconds (CDT local time: July 21, 2023 12:30:00 to August 04, 2023 12:00:00).
- **Synchronized 10 Hz Streams:** `main_data/immu`, `sub_data/head_direction`, and `main_data/pressure` share identical timestamps at 0.1-second intervals (864,000 records/day).
- **Ground-Truth Label Synchronization:** `behavior_labels/individual` is sampled at 1 Hz for July 25, 2023 (86,400 records/cow). Every 1-second behavior label aligns with exactly ten 10 Hz IMU sensor samples ($t, t+0.1, \dots, t+0.9$).

---

## 5. Target Labels & Class Distributions

### 5.1 Primary Supervised Target: Ethological Behavior (July 25, 2023)
Annotated from synchronized video over a 24-hour gold-standard benchmark period for all cows.

| Code | Class Label | Biological Definition | Overall Frequency (C01–C10) | Compatible Neckband Dynamics |
| :---: | :--- | :--- | :---: | :--- |
| **0** | Unknown / Transition | Head occluded or transition state | 151,984 (17.59%) | Unannotated background |
| **1** | Walking | Active locomotion | 10,922 (1.26%) | High `accel_norm` dynamic variance |
| **2** | Standing | Stationary upright posture | 192,081 (22.23%) | Intermediate pitch (~ -35°), low variance |
| **3** | Feeding (Head Up) | Chewing/masticating with elevated head | 52,240 (6.05%) | Pitch ~ -30° to -35°, cyclic oscillation |
| **4** | Feeding (Head Down)| Grazing/bunk feeding with lowered head | 80,053 (9.27%) | Pitch ~ -50° to -72° (head down) |
| **5** | Licking | Social/self grooming | 6,462 (0.75%) | High roll angle deflection |
| **6** | Drinking | Water intake at trough | 8,918 (1.03%) | Static lowered head posture |
| **7** | Lying | Resting sternal/lateral recumbency | 361,340 (41.82%) | Horizontal pitch (~ -9° to 0°), minimal motion |

### 5.2 Posture Target: Lying vs Standing (15-Day Continuous)
Recorded by ankle pedometers (`main_data/ankle/`) for cows C01–C10 at 1-minute resolution across all 15 days (197,920 records). Provides continuous ground truth for resting/lying duration vs active time.

### 5.3 Health Events & Disease Labels
An analysis of `sub_data/health_records` across all 16 cows revealed that **no new acute clinical disease diagnoses occurred during the 14-day trial period (July 21 – August 04, 2023)**. Cows were healthy active participants in an experimental research protocol (`onexp - 683CC`). Routine pregnancy checks and pen moves took place, and Cow C07 had pneumonia/lameness weeks prior on July 2–3. Consequently, acute disease diagnosis cannot be used as an instantaneous supervised target for this 14-day collar sensor dataset.

---

## 6. Recommended Action Summary

- **Retain for Training Dataset (KEEP):**
  - Neck IMU Accelerometer (`main_data/immu/T01..T10/` without magnetometer columns).
  - Derived Head Angles (`sub_data/head_direction/T01..T10/` with pitch, roll, accel_norm; drop yaw).
  - Barn Ambient THI, Temperature & Humidity (`main_data/thi/`).
  - Collar Microclimate Temperature (`sub_data/neck_dev_temp/T01..T10/`).
  - Ground-Truth Behavior Labels (`behavior_labels/individual/C01_0725..C10_0725.csv`).
  - Ankle Lying Ground Truth (`main_data/ankle/C01..C10/`).
  - Veterinary Health Metadata (`sub_data/health_records/C01..C16.csv`).

- **Exclude from Training (Preserve in Research Archive):**
  - `main_data/cbt` (Core Body Temp).
  - `main_data/pressure` (Barometric Pressure & Elevation).
  - `main_data/uwb` and `sub_data/uwb_distance` (Indoor UWB Positioning & Ranging).
  - `sub_data/ankle_accel` (Leg Acceleration).
  - `main_data/milk` (Milking Parlor Yields).
  - `sub_data/visual_location` (Computer Vision Localization).
  - `main_data/weather` (External Weather Station Files).
  - Reference Tags T13 & T14 (Stationary non-animal nodes).
  - Unmatched Visual Labels C11–C16 (No collar sensor data).
