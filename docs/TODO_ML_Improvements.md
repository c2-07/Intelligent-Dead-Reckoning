# Pending ML Experiments (To Be Tested Later)

## Task: Trajectory Loss + ZUPT Experiment
**Branch:** `experiment-trajectory-loss`
**Script:** `scripts/train_trajectory.py`

### Objective
Attempt to break the 11.52% drift limit by modifying the Loss Function. Currently, the AI minimizes **instantaneous speed error**. This experiment changes the loss function to minimize **accumulated distance error (Trajectory Loss)** over the 5-second window. This mathematically forces the network to self-correct its biases. 

It also includes **ZUPT (Zero-Velocity Update)** logic, which overrides the AI and forces the speed to `0.0 km/h` if the accelerometer variance implies the car is idling at a red light.

### Why it is paused
1. **GitHub LFS Bandwidth Limit:** The source dataset repository (`onyekpeu/IO-VNBD`) exhausted its GitHub LFS bandwidth limit. Local pulls return 1kb pointer files instead of the 2GB CSV files.
2. **Colab Quota Exhaustion:** We hit the `TooManyAssignmentsError` GPU limit on Google Colab.

### How to resume
Once the Colab quota resets (or using a different Google account), run the following on a fresh Colab T4 GPU:
```bash
git clone https://github.com/c2-07/Intelligent-Dead-Reckoning-SIH.git
cd Intelligent-Dead-Reckoning-SIH
git checkout experiment-trajectory-loss

# The script handles the git lfs pull automatically
uv run python scripts/train_trajectory.py
```
If the evaluation proves successful, export the resulting `.pth` to `.onnx` and replace the `champion_model.onnx` in the Flutter App.
