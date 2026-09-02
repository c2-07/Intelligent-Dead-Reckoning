# Session Handover Context: Machine Learning to Mobile App Transition

## 1. Current State of the Project
The Machine Learning pipeline for the ISRO Deep Inertial Odometry (Dead Reckoning) project is **100% complete and production-ready**. 
The repository has been fully refactored from messy exploratory scripts into a professional, modular Python package. 

### The Champion Model
* **Architecture:** ResNet-1D + BiLSTM (`RoNIN_ResNet_LSTM`)
* **Performance:** 11.52% Distance Drift over 1km GPS blackouts.
* **Weights:** 
  * `models/resnet_bilstm_v1.pth` (PyTorch training weights)
  * `models/resnet_bilstm_v1.onnx` (Optimized for Android/Flutter Edge Inference)

## 2. Repository Layout (Refactored)
* `src/dead_reckoning/`: Contains the modular ML source code (`model.py` and `dataset.py`).
* Root Scripts: `train.py`, `evaluate.py`, `export_model.py`, and `visualize.py` are highly lightweight entry points that import from `src`.
* `pyproject.toml`: Uses `uv` for lightning-fast, docker-free dependency management (requires `uv sync`).

## 3. Important Context & Gotchas for Next Session
* **Dataset/LFS Block:** The external IO-VNBD dataset repository currently has an exhausted Git LFS bandwidth limit. Running `git lfs pull` locally will fail and only download 1kb text pointers.
* **CRITICAL WARNING:** **Do NOT run `uv run python train.py` locally!** Because of the LFS block, the script will train on empty/corrupted data and overwrite the golden `resnet_bilstm_v1.pth` model. If you need to retrain, do it on Google Colab after quota resets.
* **Pending ML Experiments:** If you want to push the drift below 10%, read `docs/TODO_ML_Improvements.md` (Trajectory Loss + ZUPT experiment).

## 4. Immediate Goal for the Next Session
**Start the Android Application.**
The user has specified that the mobile application will live in a **separate repository**. 

**The Flutter App needs to replicate the Python inference pipeline in Dart:**
1. Collect Accelerometer & Gyroscope data at ~10Hz (50-frame rolling window).
2. Apply the mathematical **Gravity Rotation Matrix** to align the phone's axes to Earth.
3. Pass the rotated buffer `(1, 6, 50)` into the `resnet_bilstm_v1.onnx` model using `onnxruntime` or `tflite_flutter` to predict forward speed.
4. Retrieve Absolute Heading from the phone's Magnetometer (Compass).
5. Perform Trigonometric Dead Reckoning (`X = X + Speed * cos(Heading) * dt`) and map the UI breadcrumbs.
