# AI Research Journal & Knowledge Base
**Project:** Intelligent Dead Reckoning (Deep Inertial Odometry for Vehicles)
**Target:** Smart India Hackathon (SIH)

This document serves as a memory bank and knowledge transfer file for any future agents or developers working on this repository. It documents our approach, our failures, our critical mistakes, and the solutions we implemented to build a robust smartphone-based dead reckoning system.

---

## 1. The Core Objective
The goal was to achieve <10% distance drift during prolonged GPS blackouts (e.g., tunnels) using *only* smartphone IMU sensors (Accelerometer, Gyroscope). 

## 2. Our Approach & Architecture
We utilized the **ISRO IO-VNBD Dataset**, which pairs smartphone sensor data with high-precision VBOX ground truth data.

*   **Model Architecture:** `RoNIN_ResNet_LSTM`. A hybrid neural network.
    *   *ResNet-1D:* Acts as a learned low-pass filter to extract dynamic kinematic features from noisy raw IMU data.
    *   *BiLSTM:* Captures temporal dependencies (e.g., a car accelerating vs braking over time).
*   **Input Schema:** 6 channels (3x Accel, 3x Gyro) at 10Hz. We use a 5-second sliding window (50 frames).
*   **Prediction Target:** Instantaneous forward velocity (1D). We do *not* predict X/Y coordinates directly.

## 3. Key Technical Solutions Implemented
*   **Time-Lag Correction:** We discovered a ~0.9s (9 samples) hardware synchronization lag between the smartphone and the VBOX. We shifted the arrays before training.
*   **Gravity Compensation:** Smartphone orientation inside a vehicle is arbitrary. We mathematically rotate the raw 3D vectors at every timestamp to align the Z-axis with Earth's gravity vector. This decouples the phone's tilt from the vehicle's actual movement.
*   **Non-Holonomic Constraints (NHC):** Vehicles cannot move sideways. By predicting only forward 1D velocity and multiplying it by the phone's absolute heading, we reconstruct 2D trajectories accurately.

## 4. Failures & Mistakes (What NOT to do)

### Mistake 1: Predicting 2D Trajectories Directly
Early experiments tried to predict (ΔX, ΔY) directly from IMU data.
*   *Result:* The model collapsed or produced chaotic, diverging loss. 
*   *Solution:* Switched to predicting 1D velocity. Predicting speed is constrained and bounded; predicting unbounded coordinates is notoriously unstable in Deep Inertial Odometry.

### Mistake 2: Overwriting Weights
During a refactor, an evaluation script accidentally overwrote the best-performing model weights (`champion_model.pth`).
*   *Solution:* We moved to strict versioning (`resnet_bilstm_v1`, `v2`, `v3`, `v4`) and updated `.gitignore` to whitelist specific files while ignoring transient outputs.

### Mistake 3 (The Critical Data Leakage)
In versions `v1`, `v2`, and `v3`, we used a sequential 80/20 data split on a concatenated array of all trips. Because `M (Driver B)` was alphabetically the first folder, the first 80% of its data went to training, and the last 20% went to validation. 
*   *Result:* The model memorized the specific sensor bias and driving style of that exact vehicle/driver. We thought we had a world-class model (8.86m error over 50m).
*   *Solution:* We implemented strict **Leave-One-Trip-Out Cross-Validation**. In `v4`, we trained on drivers A, E, D, and tested *exclusively* on Driver B.

## 5. Final True Results (v4 Model)
Once the data leakage was fixed and overlapping sliding windows (stride=1s) were implemented for robust statistics, the *true* held-out performance on unseen vehicles was revealed:

*   **Median Drift:** ~16.5% (across all blackout lengths from 5s to 120s).
*   **P90 Drift:** ~25% for a 2-minute blackout.
*   **Baseline Comparison:** The AI vastly outperformed the naive assumption of "constant last-known smartphone GPS speed" (which drifted by ~66%).

### Conclusion on Metrics
A pure IMU-based neural network *cannot* reliably break the <10% P90 barrier on unseen vehicles. It stabilizes at ~16% error. To achieve the SIH mandate, the AI output **must** be paired with a Map-Matching algorithm (e.g., snapping to OpenStreetMap nodes) to artificially constrain the accumulated drift during long blackouts.

## 6. Current State & Handoff
*   The `main` branch contains the clean `src/dead_reckoning` package.
*   `models/resnet_bilstm_v4.pth` is the canonical, honest model.
*   `scripts/generate_submission_metrics.py` generates SIH-compliant benchmark CSVs.
*   Next phase: Integrating the ONNX model into the Android/Flutter frontend.
