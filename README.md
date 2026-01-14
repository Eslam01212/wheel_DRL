# Self-Supervised Wheel Deformation Prediction for Robust DRL Navigation (ROS2/Gazebo)

This repository contains the reference implementation for the paper **“Self-Supervised Wheel Deformation Prediction for Robust Deep RL Navigation in Harsh Environments.”**

The code provides:
- A self-supervised **wheel deformation predictor** (MLP) trained from collected proprioceptive/action-history data.
- A ROS2/Gazebo **Gymnasium-style navigation environment** where the predictor can be enabled/disabled.
- Training of DRL navigation policies **with prediction** and **without prediction** (ablation).
- A unified evaluation script to compare both policies under identical conditions.

---

## Repository entry points

### Main runner (training)
- **`ZMAIN.py`**  
  Runs training pipelines (with prediction and without prediction).

### Evaluation (testing / comparison)
- **`eval_compare.py`**  
  Evaluates trained checkpoints and prints metrics comparison.

---

## Repository structure (main files)

### Training / experiments
- `ZMAIN.py` — orchestrates training runs
- `d.py` — PPO training script called by `ZMAIN.py`

### Predictor (wheel deformation)
- `collect_datasets.py` — collects `.npz` dataset for predictor training
- `train_wheel.py` — trains predictor and exports `wheel_predictor.pt`
- `predictor.py` — predictor model + runtime inference

### Environment (ROS2/Gazebo)
- `env_teacher.py` — Gym environment wrapper
- `env_core.py` — ROS2 core (I/O, observation assembly, predictor hook)
- `env_utils.py` — utilities (normalization, reward helpers, etc.)

---

## Requirements

### System
- Ubuntu + **ROS 2** + **Gazebo**
- A working simulation launch for your robot that publishes the topics required by the environment.

### Python
Typical dependencies:
- `numpy`, `torch`
- `gymnasium`
- `stable-baselines3`
- `rclpy`
- optional: `opencv-python` (if image/depth observations are used)

> Run everything from a terminal where ROS2 and your workspace are sourced.

---

## Setup

1) Source ROS2 and your workspace:
```bash
source /opt/ros/<ros_distro>/setup.bash
source <your_ws>/install/setup.bash
```

2) Launch the simulator / navigation stack (separate terminal), as you normally do.

---

## Stage 1 — Train wheel predictor (optional)

Skip this section if you already have `wheel_predictor.pt`.

### 1) Collect predictor dataset
```bash
python3 collect_datasets.py
```

### 2) Train predictor
```bash
python3 train_wheel.py
```

Expected output:
- `wheel_predictor.pt`

---

## Stage 2 — Train DRL navigation policies (with and without prediction)

Run the full training pipeline:
```bash
python3 ZMAIN.py
```

This trains two policies:
- **Baseline (without prediction / ablation)**
- **Predictor-augmented (with prediction)**

Checkpoints are saved according to your configuration (commonly as `.zip` files).

---

## Evaluation — Compare trained policies

Run:
```bash
python3 eval_compare.py
```

`eval_compare.py` loads two checkpoints (paths are set inside the script; commonly under `./compare/`), for example:
- `./compare/ppo_no_predictor.zip`
- `./compare/ppo_with_predictor.zip`

If your filenames/paths differ, update them inside `eval_compare.py`.

### Reported metrics
Evaluation reports:
- **SR**: success rate (higher is better)
- **ST**: mean number of steps over successful episodes (lower is better)
- **SL**: mean cumulative slip indicator over successful episodes (lower is better)
- **PI**: mean cumulative pitch magnitude over successful episodes (lower is better)
- **RI**: mean cumulative roll magnitude over successful episodes (lower is better)

---

## How “with vs without prediction” works

- When enabled, the predictor outputs are appended to the policy observation.
- The baseline disables predictor influence using an internal flag (e.g., zeroing predictor outputs) while keeping all other settings identical.
- This provides a fair ablation under the same simulator, sensing suite, and termination conditions.

---

## Citation

If you use this repository, please cite:

**Self-Supervised Wheel Deformation Prediction for Robust Deep RL Navigation in Harsh Environments**

(Add BibTeX here.)

---

## Contact

Eslam Mohamed — FEUP / University of Porto
