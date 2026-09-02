# Intelligent Dead Reckoning (SIH 2026 - Problem 26168)

An analytical Deep Learning solution for vehicle navigation in GPS-denied environments (e.g., long tunnels, urban canyons). This system achieves sub-20% distance drift rates over 1km utilizing smartphone IMU sensors.

## Architecture and Methodology
When a vehicle loses GPS signal, the system relies on the smartphone's internal inertial sensors:
1. **Inputs:** Raw Accelerometer (linear acceleration) and Gyroscope (angular velocity) data.
2. **Deep Learning Core:** A ResNet-BiLSTM neural network processes the time-series data over a 5-second sliding window to predict the vehicle's forward velocity.
3. **Sensor Fusion:** The predicted forward velocity is fused with the smartphone's hardware Magnetometer (absolute compass heading) to mitigate quadratic steering drift associated with raw gyroscope integration.
4. **Outputs:** Real-time Cartesian (X, Y) trajectory mapping of the vehicle.

## Pre-trained Models

| Model | File | 50m Avg Error | 1km Avg Error | 50m Drift | 1km Drift |
|---|---|---|---|---|---|
| **v2 (Latest)** | `models/resnet_bilstm_v2.pth` | 10.11m | 135.32m | 19.90% | 13.52% |
| v1 (Original) | `models/resnet_bilstm_v1.pth` | 11.52m | 177.56m | 22.70% | 17.75% |

Both models use the same `RoNIN_ResNet_LSTM` architecture. The v2 model was trained as a reproducibility test using identical code and hyperparameters, demonstrating that the training pipeline produces consistent results across runs.

## Research and Optimization
Empirical testing of multiple architectures was conducted on the ISRO `IO-VNBD` dataset. The experiments are documented across the branches of this repository:

* **`main` (Current Model):** An unconstrained `ResNet-BiLSTM`. Selected for its mathematical robustness and resistance to overfitting on small datasets.
* **`experiment-reproduce-benchmark`:** Reproducibility test confirming the training pipeline produces consistent drift rates (10-12m over 50m segments).
* **`experiment-trajectory-loss`:** Experimental Trajectory Loss function that minimizes accumulated distance error over a 5-second window, combined with ZUPT (Zero-Velocity Update) heuristics.
* **`experiment-sota-2025`:** Implemented a DUET Dynamic Bias network, Ackermann Kinematic constraints ($a_y = v_x \times \omega_z$), and a Low-Pass Suspension Decoupler. Demonstrated that over-constraining the loss function on a dataset of this size leads to a constraint bottleneck (13.6% drift).
* **`experiment-champion-boost`:** Applied a data overlap strategy (5x multiplier). Achieved lower training error but suffered from overlapping data leakage during validation (13.7% drift).

**Conclusion:** The empirical data suggests that ~10-12m position error over 50m segments approaches the information-theoretic limit for this specific hardware and dataset configuration. To eliminate the remaining error, this model is designed to act as the core inference engine for a Map-Matching application, which constrains the trajectory to known road networks (e.g., OpenStreetMap).

## Repository Structure
* `scripts/train.py`: The training pipeline. Downloads the IO-VNBD dataset, applies lag correction, and trains the ResNet-BiLSTM model.
* `models/resnet_bilstm_v1.pth`: Pre-trained weights (v1, original training run).
* `models/resnet_bilstm_v2.pth`: Pre-trained weights (v2, reproducibility test).
* `models/resnet_bilstm_v1.onnx`: ONNX export optimized for Android/Flutter edge inference.
* `scripts/evaluate.py`: The benchmark script. Simulates 50m and 1km GPS blackouts across the validation set and calculates exact drift rates using Sensor Fusion.
* `scripts/visualize.py`: Generates a spatial plot (`blackout_simulation.png`) comparing the model's dead reckoning trajectory against the true GPS trajectory.
* `src/dead_reckoning/dataset.py`: Contains dataset preprocessing logic, including time lag correction and gravity orientation alignment.

## Usage Instructions

### Prerequisites
Python must be installed. The `uv` package manager is recommended for execution.

### 1. Run the Evaluation Benchmark
To test the pre-trained model and compute the drift metrics:
```bash
# Using UV (automatically handles dependencies)
uv run scripts/evaluate.py

# Standard pip
pip install -e .
python scripts/evaluate.py
```
*Note: The script automatically downloads the required CSV datasets from the ISRO repository if they are not present locally.*

### 2. Visualize a GPS Blackout
To generate a 2D spatial plot of a 60-second GPS blackout:
```bash
uv run scripts/visualize.py
# or: python scripts/visualize.py
```
This outputs a `blackout_simulation.png` file comparing the True Path to the Fused Path.

### 3. Train the Model
To train the `ResNet-BiLSTM` model from scratch:
```bash
uv run scripts/train.py
# or: python scripts/train.py
```

### 4. Running on Google Colab
You can easily train and evaluate this model using a free GPU on Google Colab:
1. Create a new notebook in [Google Colab](https://colab.research.google.com/).
2. Change the runtime to GPU (**Runtime** > **Change runtime type** > **T4 GPU**).
3. Paste and run the following snippet in a cell:

```python
# Clone the repository
!git clone https://github.com/c2-07/Intelligent-Dead-Reckoning-SIH.git
%cd Intelligent-Dead-Reckoning-SIH

# Pull the ISRO dataset via Git LFS
!git lfs install
!git lfs pull

# Setup the python environment with `uv`
!pip install uv
!uv pip install -e . --system

# Train the model (saves to models/resnet_bilstm_latest.pth)
!uv run scripts/train.py

# Evaluate the newly trained model
!uv run scripts/evaluate.py --model models/resnet_bilstm_latest.pth

# Or evaluate the pre-trained v2 model
!uv run scripts/evaluate.py --model models/resnet_bilstm_v2.pth
```
