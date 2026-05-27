
raw
Readme · MD
# The Head Counter — Multi-Modal Library Occupancy Detection System
 
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=ultralytics)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![Fusion](https://img.shields.io/badge/Fusion-3--Channel-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
 
**A real-time, multi-sensor occupancy detection system that fuses camera, seat sensor, and CO2 data to estimate how many people are inside a library — no manual counting needed.**
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Detection Channels](#detection-channels)
- [Fusion Logic](#fusion-logic)
- [Occupancy Status Tiers](#occupancy-status-tiers)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Sample Output](#sample-output)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
---
 
## Overview
 
Manual library headcounts are slow, inaccurate, and impractical at scale. **The Head Counter** solves this by combining three independent sensing modalities — computer vision, physical load cells, and CO2 air quality — into a single fused occupancy estimate, updated in real time.
 
Each channel compensates for the other's blind spots:
 
| Challenge | Solution |
|---|---|
| Camera can't see people behind walls | CO2 fills in the gaps |
| CO2 doesn't tell you exact seat count | Load cells track every chair |
| Load cells miss standing occupants | YOLO camera covers open areas |
| Single-sensor failures cause wrong estimates | Weighted fusion reduces noise |
 
**Result:** A robust, production-ready occupancy estimate with a live capacity percentage and actionable status (OPEN / BUSY / FULL).
 
---
 
## System Architecture
 
```
┌─────────────────────────────────────────────────────────┐
│               THE HEAD COUNTER — PIPELINE               │
└─────────────────────────────────────────────────────────┘
 
  ┌────────────────┐   ┌────────────────┐   ┌─────────────────┐
  │  CHANNEL 1     │   │  CHANNEL 2     │   │  CHANNEL 3      │
  │  Camera        │   │  Seat Sensors  │   │  CO2 Sensor     │
  │  (YOLOv8)      │   │  (Load Cell)   │   │  (RandomForest) │
  └───────┬────────┘   └───────┬────────┘   └────────┬────────┘
          │                    │                      │
          ▼                    ▼                      ▼
   Person count         Occupied seats          Room status
   from image           from voltage             (binary)
          │                    │                      │
          └────────────────────┼──────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  WEIGHTED FUSION │
                    │  w = [0.25,      │
                    │       0.55,      │
                    │       0.20]      │
                    └────────┬─────────┘
                             │
                             ▼
                   Fused Occupancy Estimate
                   Capacity % + Status Label
```
 
---
 
## Detection Channels
 
### Channel 1 — Camera (YOLOv8)
 
A custom-trained YOLOv8 model performs real-time person detection on camera frames.
 
- **Model:** YOLOv8 fine-tuned on library/indoor datasets
- **Confidence threshold:** 0.30
- **Output:** Raw person count from image
- **Weight in fusion:** 25%
The camera channel excels at detecting people in open spaces but can be obstructed by furniture, pillars, or camera angle.
 
---
 
### Channel 2 — Seat Sensors (Load Cell + Linear Regression)
 
Each seat has a load cell sensor that outputs a voltage proportional to applied weight. A trained `LinearRegression` model maps raw voltage to estimated weight.
 
- **Occupied threshold:** Predicted weight > 20 kg (scaled)
- **Voltage range:** 0.0–0.8 V (empty) → 2.0–3.3 V (occupied)
- **Seats monitored:** Up to 50
- **Model:** `sensor_model.pkl` (scikit-learn Linear Regression)
- **Weight in fusion:** 55%
The seat sensor channel is the highest-weighted channel because it directly measures physical occupancy at every individual seat, making it the most reliable ground truth.
 
---
 
### Channel 3 — CO2 Sensor (Random Forest Classifier)
 
Human breathing raises ambient CO2 levels. A trained `RandomForestClassifier` takes three live air quality readings and predicts whether the room is occupied or empty.
 
- **Features:** `CO2_smooth (ppm)`, `CO2_change (ppm/step)`, `HumidityRatio (kg/kg)`
- **Output:** Binary classification — OCCUPIED or EMPTY
- **Model:** `co2_model.pkl` (scikit-learn RandomForest)
- **Derived estimate:** 25 people (50% capacity) if OCCUPIED, else 0
- **Weight in fusion:** 20%
CO2 captures occupants that cameras can't see (e.g., people in enclosed reading rooms or basement sections).
 
---
 
## Fusion Logic
 
The three channel outputs are combined using a fixed weighted average:
 
```python
weights    = [0.25, 0.55, 0.20]   # camera, sensor, co2
raw_counts = [yolo_count, sensor_count, co2_count]
fused      = int(round(np.dot(weights, raw_counts)))
```
 
The seat sensor carries the highest weight (55%) because it provides direct, per-seat physical measurement. The camera (25%) adds spatial context, and the CO2 (20%) acts as a sanity check for areas outside camera coverage.
 
---
 
## Occupancy Status Tiers
 
| Status | Threshold | Label |
|---|---|---|
| 🟢 OPEN | < 60% capacity | Plenty of seats available |
| 🟡 BUSY | 60% – 89% capacity | Limited seats available |
| 🔴 FULL | ≥ 90% capacity | Library is at capacity |
 
This tiered output is designed to be displayed on a public-facing dashboard, entrance display, or mobile app so students can check availability before travelling to the library.
 
---
 
## Project Structure
 
```
your-repo/
├── demo.py                  ← Full fusion pipeline (entry point)
├── weight/
│   └── best.pt              ← YOLOv8 trained weights
├── MODELS/
│   ├── sensor_model.pkl     ← Load cell linear regression model
│   └── co2_model.pkl        ← CO2 random forest classifier
└── sample_images/
    └── test.jpg             ← Test image for YOLO inference
```
 
---
 
## How to Run
 
### Prerequisites
 
- Python 3.10 or above
- pip
### 1. Clone the repository
 
```bash
git clone https://github.com/your-username/head-counter.git
cd head-counter
```
 
### 2. Install dependencies
 
```bash
pip install ultralytics scikit-learn pandas joblib numpy
```
 
### 3. Add model files
 
Place the following files in their respective folders before running:
 
```
weight/best.pt
MODELS/sensor_model.pkl
MODELS/co2_model.pkl
sample_images/test.jpg
```
 
### 4. Run the demo
 
```bash
# Default (uses sample_images/test.jpg)
python demo.py
 
# Custom image
python demo.py --image path/to/your/image.jpg
```
 
---
 
## Sample Output
 
```
=======================================================
   The Head Counter — Multi-Modal Library Occupancy Detection
=======================================================
 
[1/3] Camera Channel (YOLOv8)...
      Image                  : sample_images/test.jpg
      People detected        : 8
 
[2/3] Seat Sensor Channel (Load Cell)...
      Seats scanned          : 50
      Occupied seats         : 29
 
[3/3] CO2 Sensor Channel (RandomForest)...
      Live reading           : CO2=650 ppm, ΔCO2=+12.5, Humidity=0.004
      Room status (model)    : OCCUPIED
      Derived occupancy est. : 25 people
 
=======================================================
          The Head Counter — Library Occupancy Report
=======================================================
  Seat Sensors  (ground truth) :  29 people
  Camera        (YOLOv8)       :   8 people
  CO2 Sensor    (estimate)     :  25 people  [OCCUPIED]
-------------------------------------------------------
  Fused Occupancy Estimate     :  22 people
  Capacity Usage               : 44%
  Status                       : OPEN — Plenty of seats available
=======================================================
```
 
---
 
## Tech Stack
 
| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Ultralytics YOLOv8 | Real-time person detection from camera |
| scikit-learn | Load cell regression + CO2 classification |
| NumPy | Weighted fusion calculation |
| Pandas | Feature formatting for model inference |
| Joblib | Model serialization and loading |
 
---
 
## Future Work
 
- **Live webcam stream** — replace static image with real-time video feed
- **MQTT integration** — connect actual IoT load cell and CO2 hardware
- **Dashboard UI** — web-based occupancy display with historical trends
- **Adaptive weights** — dynamically adjust fusion weights based on sensor confidence scores
- **Alert system** — push notification when capacity exceeds 80%
- **Multi-zone support** — extend to track occupancy per floor or room section
---
 
## Author
 
Built with a focus on practical, deployable multi-modal sensing — not just a single-model demo.
 
Made with Python • YOLOv8 • scikit-learn • NumPy
