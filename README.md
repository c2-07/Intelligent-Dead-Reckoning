# Intelligent Dead Reckoning (SIH 2026 - Problem 26168)

An analytical Deep Learning solution for vehicle navigation in GPS-denied environments (e.g., long tunnels, urban canyons). This system achieves an 11.5% distance drift rate over 1km utilizing smartphone IMU sensors.

## Architecture and Methodology
When a vehicle loses GPS signal, the system relies on the smartphone's internal inertial sensors:
1. **Inputs:** Raw Accelerometer (linear acceleration) and Gyroscope (angular velocity) data.
2. **Deep Learning Core:** A ResNet-BiLSTM neural network processes the time-series data over a 5-second sliding window to predict the vehicle's forward velocity.
3. **Sensor Fusion:** The predicted forward velocity is fused with the smartphone's hardware Magnetometer (absolute compass heading) to mitigate quadratic steering drift associated with raw gyroscope integration.
4. **Outputs:** Real-time Cartesian (X, Y) trajectory mapping of the vehicle.

## Research and Optimization
Empirical testing of multiple architectures was conducted on the ISRO `IO-VNBD` dataset. The experiments are documented across the branches of this repository:

* **`main` (Current Model):** An unconstrained `ResNet-BiLSTM`. Achieved 11.52% drift on the validation set. Selected for its mathematical robustness and resistance to overfitting on small datasets.
* **`experiment-sota-2025`:** Implemented a DUET Dynamic Bias network, Ackermann Kinematic constraints ($a_y = v_x \times \omega_z$), and a Low-Pass Suspension Decoupler. Demonstrated that over-constraining the loss function on a dataset of this size leads to a constraint bottleneck (13.6% drift).
* **`experiment-champion-boost`:** Applied a data overlap strategy (5x multiplier). Achieved lower training error but suffered from overlapping data leakage during validation (13.7% drift).

**Conclusion:** The empirical data suggests that 11.5% drift approaches the information-theoretic limit for this specific hardware and dataset configuration. To eliminate the remaining error, this model is designed to act as the core inference engine for a Map-Matching application, which constrains the trajectory to known road networks (e.g., OpenStreetMap).

## Repository Structure
* `scripts/train.py`: The training pipeline. Downloads the IO-VNBD dataset, applies lag correction, and trains the ResNet-BiLSTM model.
* `models/resnet_bilstm_v1.pth`: Pre-trained weights for the 11.5% drift model.
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
