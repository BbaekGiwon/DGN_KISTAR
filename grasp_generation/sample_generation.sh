#!/bin/bash
# Sample grasp generation for the KISTAR hand.
# Generates grasps for ALL objects, distributed across the listed GPUs.
# Edit the variables below, then:  bash sample_generation.sh
set -e
cd "$(dirname "$0")"

# ---- config ----
GPUS=0,1,2,3                      # GPUs to use (objects are split across them)
SEED=42                          # change the seed to accumulate more (distinct) grasps
RESULT=../data/graspdata_s$SEED  # output dir (per seed)
MESH=../data/meshdata_norm       # normalized object meshes
export OBJ_SCALE=0.03,0.045,0.06,0.075   # per-grasp size = bounding-sphere radius (m) -> 6/9/12/15 cm
# ----------------

export CUDA_VISIBLE_DEVICES=$GPUS
N_GPU=$(echo $GPUS | tr ',' '\n' | wc -l)
BATCH=1440                                  # grasps per object
MAXTOTAL=$(( BATCH * N_GPU ))               # concurrent batch over all GPUs

python scripts/generate_grasps_kistar.py --all --seed $SEED \
  --data_root_path $MESH --result_path $RESULT \
  --batch_size_each $BATCH --max_total_batch_size $MAXTOTAL --n_iter 6000 \
  --temperature_decay 0.98 --switch_possibility 0.1 \
  --w_pen 100 --w_spen 20 --w_joints 20
# NOTE: no --overwrite -> re-running resumes (already-generated objects are skipped).
#       Loop over several seeds to grow the dataset.

echo "done: $(ls $RESULT | wc -l)/78 objects -> $RESULT"
