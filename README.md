# DGN_KISTAR

A dexterous-grasp **dataset-generation pipeline for the KISTAR 16-DoF hand**.
- Original code: [DexGraspNet](https://arxiv.org/abs/2210.02697)
- Initial idea by **[Chanyoung Ahn](https://github.com/cold-young)** (PRIME LAB) - [DexGrasp_KIST](https://github.com/cold-young/DexGrasp_KIST)
- Modified by **[Giwon Baek](https://github.com/BbaekGiwon)** (HARI LAB), 2026.06 - Adjust DexGraspNet to KISTAR hand

---
## KISTAR config
[KISTAR Configuration Document](./assets/DGN_KISTAR.pdf)

The KISTAR hand has **16 DoF** — thumb / index / middle / ring, each with 4 joints
(`*_joint_0` = abduction/opposition, `*_joint_1..3` = flexion).

## Prerequisites
- Linux + NVIDIA GPU (CUDA 11.7), conda
- **A single environment runs generation, validation, and visualization** (Python 3.8 / torch 2.0.1+cu117)
- Isaac Gym is downloaded separately from NVIDIA (see below)

## Get Started

### 1. Create Python Environment
```bash
conda env create -f environment.yml       # creates env "dgn_kistar" (python 3.8 + requirements.txt)
conda activate dgn_kistar
# or manually:
#   conda create -n dgn_kistar python=3.8 -y && conda activate dgn_kistar
#   pip install -r requirements.txt        # torch+cu117, viser, trimesh, ...
```

### 2. Install Isaac Gym Environment
```bash
# Download Isaac Gym Preview 4 from developer.nvidia.com/isaac-gym, then unzip
export CUDA_HOME=$CONDA_PREFIX; export PATH=$CUDA_HOME/bin:$PATH
cd <isaacgym>/python && pip install -e . && cd -
```
> ⚠️ In code, import `isaacgym` **before** `torch` (the validation scripts already do this).

### 3. Install Packages (thirdparty)

These are compiled CUDA extensions (not on PyPI). Set the CUDA env vars first:
```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

| package | role | required |
|---|---|---|
| `pytorch_kinematics` | forward kinematics (**bundled in this repo**) | ✅ |
| `TorchSDF` | signed-distance for `E_pen` / `E_dis` | ✅ |
| `pytorch3d` | surface / point sampling | ✅ |
| Isaac Gym | physics validation | ✅ (see step 2) |
| `ManifoldPlus` | watertight mesh repair | ⬜ asset prep only |
| `CoACD` | approximate convex decomposition | ⬜ asset prep only |

**Required** (generation + validation) — run from the repo root:
```bash
# pytorch_kinematics — bundled under thirdparty/
cd thirdparty/pytorch_kinematics && pip install -e . && cd ../..

# pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git thirdparty/pytorch3d
cd thirdparty/pytorch3d && pip install -e . && cd ../..

# TorchSDF
git clone https://github.com/wrc042/TorchSDF.git thirdparty/TorchSDF
cd thirdparty/TorchSDF && git checkout 0.1.0 && bash install.sh && cd ../..
```

**Optional — asset processing only** (preparing *new* object meshes; skip if you use the provided `meshdata_norm`):
```bash
# ManifoldPlus (watertight repair)
git clone https://github.com/hjwdzh/ManifoldPlus.git thirdparty/ManifoldPlus
cd thirdparty/ManifoldPlus && git submodule update --init --recursive
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && cd ../../..

# CoACD (convex decomposition)
git clone --recurse-submodules https://github.com/SarahWeiii/CoACD.git thirdparty/CoACD
cd thirdparty/CoACD && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && cd ../../..
```

Verify the unified env:
```bash
python -c "from isaacgym import gymapi; import torch, pytorch3d, torchsdf, pytorch_kinematics, viser; print('OK: gen + val + viz in one env')"
```

## Data Structure

```
data/
+-- meshdata_norm/<code>/coacd/{decomposed.obj, coacd.urdf}   # normalized mesh (unit bounding sphere) + Isaac urdf
+-- graspdata_v5_s<seed>/<code>.npy                            # generated grasps (pre-validation)
+-- dataset_v5_s<seed>/<code>.npy                              # Isaac-validated grasps
+-- dataset_v5_merged/<code>.npy                               # final dataset (merged over seeds)
```
- 78 objects (50 contactdb + 28 ycb); object code = `<category>-<name>`
- Each grasp entry: `{qpos, scale, E_fc, E_dis, E_pen, E_spen, E_joints}`

## How to use?
```
[normalized mesh] --generate--> graspdata --validate (Isaac)--> dataset --merge--> final dataset
```

### Object scale normalization
Each mesh is normalized to a **unit bounding sphere** (centroid at origin, max radius = 1).
At generation, every grasp randomly samples a physical size from `OBJ_SCALE`
(the bounding-sphere **radius** in meters):
```bash
export OBJ_SCALE=0.03,0.045,0.06,0.075     # = diameters 6 / 9 / 12 / 15 cm
```
This lets even large objects fit the hand's span (zero-yield objects: 10 → 0).

### Generation
Easiest — edit the variables at the top of the sample script and run it:
```bash
cd grasp_generation
bash sample_generation.sh          # all objects, multi-GPU; set GPUS / SEED inside
```
Or call the script directly:
```bash
cd grasp_generation
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OBJ_SCALE=0.03,0.045,0.06,0.075
python scripts/generate_grasps_kistar.py --all --seed 42 \
  --data_root_path ../data/meshdata_norm --result_path ../data/graspdata_s42 \
  --batch_size_each 1440 --max_total_batch_size 5760 --n_iter 6000 \
  --temperature_decay 0.98 --switch_possibility 0.1 \
  --w_pen 100 --w_spen 20 --w_joints 20
```
- Energy: `E = E_fc + 100·E_dis + 100·E_pen + 20·E_spen + 20·E_joints`
- Canonical init pose: thumb opposition (j0 = 81°, j1 = −72°), index/ring abduction ∓12° (set ~1σ inside the joint limits)
- Omit `--overwrite` to **resume** (already-generated objects are skipped).

**Grow the dataset by varying the seed.** Each seed is an independent pass over all 78 objects
(different random init → distinct grasps). Run several seeds into separate dirs, then validate &
merge them:
```bash
for SEED in 42 43 44 45; do
  python scripts/generate_grasps_kistar.py --all --seed $SEED \
    --data_root_path ../data/meshdata_norm --result_path ../data/graspdata_s$SEED \
    --batch_size_each 1440 --max_total_batch_size 5760 --n_iter 6000 \
    --temperature_decay 0.98 --switch_possibility 0.1 --w_pen 100 --w_spen 20 --w_joints 20
done
```
> The released **`dataset_v5_merged`** (2,470 valid grasps) was built from **7 passes, seeds `789, 790, 791, 792, 793, 794, 795`**.

### Validation
Easiest — edit the variables at the top of the sample script and run it:
```bash
cd grasp_generation
bash sample_validation.sh          # all objects, round-robin over GPUs; set GPUS / dirs inside
```
Or per object:
```bash
python scripts/validate_grasps_kistar.py --gpu 0 \
  --mesh_path ../data/meshdata_norm --grasp_path ../data/graspdata_s42 \
  --result_path ../data/dataset_s42 --object_code contactdb-apple \
  --penetration_threshold 0.01 --joints_threshold 0.001 --spen_threshold 0.005
```
- `valid = (E_pen < 0.01 & E_joints < 0.001 & E_spen < 0.005)  AND  (Isaac holds under 6 gravity directions)`
- A cheap energy prefilter (stored values) is applied first; the expensive Isaac sim runs only on survivors.

### Visualization
```bash
# 1) precompute a viewer bundle (hand + object mesh per valid grasp)
python tools/precompute_valid.py --dataset_dir ../data/dataset_v5_merged \
  --mesh_path ../data/meshdata_norm --out ../data/vis_valid_v5.npy
# 2) launch viser (same env)
python tools/viser_valid.py --bundle ../data/vis_valid_v5.npy --host 127.0.0.1 --port 8080
# open http://127.0.0.1:8080 ; click the slider, then use the ← / → arrow keys to scroll grasps
```

---
## License
Based on [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) (Wang et al., 2022),
licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
