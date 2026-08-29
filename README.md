# AI-Based Intelligent Dead Reckoning (SIH Proposal)

An AI/ML based Intelligent Dead Reckoning navigation system for smartphone-based navigation during GPS outages. Built for the Smart India Hackathon (SIH) problem statement by ISRO.

## The Problem
When vehicles enter tunnels, dense urban canyons, or areas with heavy tree cover, GPS signals drop or become highly inaccurate. Standard smartphone sensors (accelerometer, gyroscope) are extremely noisy due to engine vibrations and potholes, meaning standard mathematical dead reckoning fails instantly (drift accumulates exponentially).

## Our Solution
We utilize a **1D Temporal Convolutional Network (TCN)** trained on synchronized vehicle ECU and smartphone IMU datasets (IO-VNBD). The AI learns to filter out mechanical noise and extract the true forward velocity of the vehicle using only smartphone sensors.

During a GPS blackout:
1. The smartphone's Gyroscope is used for continuous heading estimation.
2. The AI model processes 2-second windows of raw IMU data (at 10Hz) to predict accurate forward velocity.
3. These are fused to simulate a highly accurate continuous trajectory until GPS is recovered.

## Repository Contents
* `preprocess_data.py` - Corrects dataset time offsets (0.9s lag) and virtually flattens phone orientation using gravity vectors.
* `train_tcn.py` - PyTorch training pipeline for the TCN model, deployed via Google Colab.
* `visualize_results.py` - Simulates a 60-second GPS blackout and plots the AI's trajectory against the ground truth and raw IMU integration.
* `best_tcn_model.pth` - The trained PyTorch model weights (RMSE ~10.5 km/h).
* `dataset_integrity_audit.md` - Comprehensive data quality report of the IO-VNBD dataset.
* `sih_dead_reckoning_plan.md` - Full system architecture and proposal plan.
