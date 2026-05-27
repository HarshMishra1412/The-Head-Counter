"""
The Head Counter — Multi-Modal Library Occupancy Detection System
demo.py
 
Runs the full fusion pipeline using actual trained models.
 
Requirements:
    pip install ultralytics scikit-learn pandas joblib numpy
 
Folder structure expected:
    your-repo/
    ├── demo.py
    ├── weight/
    │   └── best.pt                  ← YOLOv8 weights
    ├── MODELS/
    │   ├── sensor_model.pkl         ← Load cell linear regression model
    │   └── co2_model.pkl            ← CO2 RandomForest classifier model
    └── sample_images/
        └── test.jpg                 ← Test image for YOLO
 
Usage:
    python demo.py
    python demo.py --image sample_images/your_image.jpg
"""
 
import numpy as np
import pandas as pd
import joblib
import argparse
import random
import os
from ultralytics import YOLO
 
# ─── ARGUMENT PARSING ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="The Head Counter Demo")
parser.add_argument(
    "--image",
    type=str,
    default="sample_images/test.jpg",
    help="Path to test image for YOLO inference"
)
args = parser.parse_args()
 
# ─── PATH CONFIGURATION ──────────────────────────────────────────
YOLO_WEIGHTS = "weight/best.pt"
SENSOR_MODEL = "MODELS/sensor_model.pkl"
CO2_MODEL    = "MODELS/co2_model.pkl"
TEST_IMAGE   = args.image
MAX_CAPACITY = 50
 
# ─── SANITY CHECK ────────────────────────────────────────────────
missing = []
for path in [YOLO_WEIGHTS, SENSOR_MODEL, CO2_MODEL, TEST_IMAGE]:
    if not os.path.exists(path):
        missing.append(path)
 
if missing:
    print("\n[ERROR] Missing files:")
    for f in missing:
        print(f"        {f}")
    print("\nSee folder structure in the docstring at the top of this file.")
    exit(1)
 
print("=" * 55)
print("   The Head Counter— Multi-Modal Library Occupancy Detection")
print("=" * 55)
 
# ─── CHANNEL 1: YOLO CAMERA ──────────────────────────────────────
print("\n[1/3] Camera Channel (YOLOv8)...")
 
yolo_model = YOLO(YOLO_WEIGHTS)
results    = yolo_model.predict(source=TEST_IMAGE, conf=0.3, save=False, verbose=False)
yolo_count = len(results[0].boxes)
 
print(f"      Image                  : {TEST_IMAGE}")
print(f"      People detected        : {yolo_count}")
 
# ─── CHANNEL 2: LOAD CELL SEAT SENSORS ───────────────────────────
print("\n[2/3] Seat Sensor Channel (Load Cell)...")
 
sensor_model = joblib.load(SENSOR_MODEL)
 
random.seed(None)   # random seed — different values every run
people_count = 0
for _ in range(MAX_CAPACITY):
    # Simulate realistic human weight in grams (40kg to 100kg = 40000 to 100000 gm)
    # Map to voltage: empty seat ~0.5V, occupied seat ~2.0-3.3V
    is_occupied = random.random() < 0.6   # 60% chance seat is occupied
    if is_occupied:
        voltage = random.uniform(2.0, 3.3)   # high voltage = heavy weight
    else:
        voltage = random.uniform(0.0, 0.8)   # low voltage = empty seat
 
    weight = sensor_model.predict(pd.DataFrame({'Voltage (V)': [voltage]}))[0]
 
    # Scale predicted weight to human scale (gm → realistic range)
    scaled_weight = abs(weight) * 50   # scale factor to bring into human range
 
    if scaled_weight > 20000:   # > 20kg = seat occupied
        people_count += 1
 
sensor_count = people_count
print(f"      Seats scanned          : {MAX_CAPACITY}")
print(f"      Occupied seats         : {sensor_count}")
 
# ─── CHANNEL 3: CO2 SENSOR ───────────────────────────────────────
print("\n[3/3] CO2 Sensor Channel (RandomForest)...")
 
co2_model = joblib.load(CO2_MODEL)
 
# Simulated live reading — replace with actual sensor values in production
# Features must match the order used during training:
#   [CO2_smooth (ppm), CO2_change (ppm/step), HumidityRatio (kg/kg)]
live_co2_reading = [[650.0, 12.5, 0.0040]]
 
co2_prediction  = co2_model.predict(live_co2_reading)[0]   # 0 or 1
co2_label       = "OCCUPIED" if co2_prediction == 1 else "EMPTY"
 
CO2_OCCUPIED_ESTIMATE = MAX_CAPACITY // 2
co2_count = CO2_OCCUPIED_ESTIMATE if co2_prediction == 1 else 0
 
print(f"      Live reading           : CO2=650 ppm, ΔCO2=+12.5, Humidity=0.004")
print(f"      Room status (model)    : {co2_label}")
print(f"      Derived occupancy est. : {co2_count} people")
 
# ─── FUSION ──────────────────────────────────────────────────────
weights    = np.array([0.25, 0.55, 0.20])   # camera, sensor, co2
raw_counts = np.array([yolo_count, sensor_count, co2_count])
fused      = int(round(np.dot(weights, raw_counts)))
 
occupancy_pct = min(100, round((fused / MAX_CAPACITY) * 100))
 
if occupancy_pct >= 90:
    status = "FULL    — Library is at capacity"
elif occupancy_pct >= 60:
    status = "BUSY    — Limited seats available"
else:
    status = "OPEN    — Plenty of seats available"
 
print("\n" + "=" * 55)
print("          The Head Counter  — Library Occupancy Report")
print("=" * 55)
print(f"  Seat Sensors  (ground truth) : {sensor_count:>3} people")
print(f"  Camera        (YOLOv8)       : {yolo_count:>3} people")
print(f"  CO2 Sensor    (estimate)     : {co2_count:>3} people  [{co2_label}]")
print("-" * 55)
print(f"  Fused Occupancy Estimate     : {fused:>3} people")
print(f"  Capacity Usage               : {occupancy_pct}%")
print(f"  Status                       : {status}")
print("=" * 55)