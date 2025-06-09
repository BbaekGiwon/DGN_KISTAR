# validate_grasps.py

## Overview
This script validates robotic grasps using the Isaac Gym simulator. It processes grasp data, calculates contact points, and evaluates the validity of grasps based on simulation results and energy thresholds.

---

## Usage

### Command-Line Arguments
The script accepts the following arguments:

| Argument                  | Type    | Default Value                                           | Description                                                                 |
|---------------------------|---------|-------------------------------------------------------|-----------------------------------------------------------------------------|
| `--gpu`                   | `int`   | `3`                                                   | GPU ID to use for simulation.                                              |
| `--val_batch`             | `int`   | `500`                                                 | Number of grasps to validate in a single batch.                            |
| `--mesh_path`             | `str`   | `../data/meshdata`                                    | Path to the directory containing object mesh data.                         |
| `--grasp_path`            | `str`   | `../data/graspdata`                                   | Path to the directory containing grasp data files.                         |
| `--result_path`           | `str`   | `../data/dataset`                                     | Path to save the validated grasp results.                                  |
| `--object_code`           | `str`   | `sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2`         | Identifier for the object being validated.                                 |
| `--index`                 | `int`   | `None`                                                | If provided, enables debug mode and validates a single grasp by index.     |
| `--no_force`              | `flag`  | `False`                                               | If set, skips force-based optimization.                                    |
| `--thres_cont`            | `float` | `0.001`                                               | Threshold for contact point validation.                                    |
| `--dis_move`              | `float` | `0.001`                                               | Distance to move contact points for optimization.                          |
| `--grad_move`             | `float` | `500`                                                 | Gradient scaling factor for optimization.                                  |
| `--penetration_threshold` | `float` | `0.001`                                               | Threshold for penetration energy to consider a grasp valid.                |

---

## Workflow

1. **Setup**:
   - Parses command-line arguments.
   - Configures the environment and GPU device.
   - Loads grasp data from the specified file.

2. **Force-Based Optimization** (if `--no_force` is not set):
   - Initializes the hand and object models using `HandModel` and `ObjectModel`.
   - Calculates contact points and normals between the hand and object.
   - Optimizes hand poses to minimize contact loss.

3. **Simulation**:
   - Initializes the Isaac Gym simulator using `IsaacValidator`.
   - If `--index` is provided, validates a single grasp in GUI mode.
   - Otherwise, validates all grasps in batches and calculates simulation results.

4. **Validation**:
   - Combines simulation results and penetration energy thresholds to determine valid grasps.
   - Saves valid grasps to the specified result path.

5. **Cleanup**:
   - Destroys the simulator instance.

---

## Key Components

### Classes and Functions
- **`IsaacValidator`**: Handles interaction with the Isaac Gym simulator. Defined in `utils/isaac_validator.py`.
- **`HandModel`**: Represents the robotic hand model. Defined in `utils/hand_model.py`.
- **`ObjectModel`**: Represents the object model. Defined in `utils/object_model.py`.

### Data Processing
- Grasp data is loaded from `.npy` files in `--grasp_path`.
- Each grasp contains:
  - `qpos`: Joint positions and hand pose.
  - `scale`: Object scale.
  - `E_pen`: Penetration energy.

### Output
- Validated grasps are saved as `.npy` files in `--result_path`.

---

## Example Usage

### Single Grasp Validation (Debug Mode)
```bash
python scripts/validate_grasps.py --gpu 0 --index 5 --object_code ddg-gd_banana_poisson_002
```
