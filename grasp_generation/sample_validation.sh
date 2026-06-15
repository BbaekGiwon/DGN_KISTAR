#!/bin/bash
# Sample Isaac-Gym validation for generated KISTAR grasps.
# Validates every object in GRASP dir, round-robin across the listed GPUs.
# Edit the variables below, then:  bash sample_validation.sh
set -e
cd "$(dirname "$0")"

# ---- config ----
GPUS=(0 1 2 3)                    # GPUs to use
GRASP=../data/graspdata_s42       # generated grasps (input)
RESULT=../data/dataset_s42        # validated grasps (output)
MESH=../data/meshdata_norm        # normalized object meshes (+ coacd.urdf)
PEN=0.01; JOINTS=0.001; SPEN=0.005   # validity thresholds (E_pen / E_joints / E_spen)
# ----------------

mkdir -p $RESULT
mapfile -t OBJS < <(ls $GRASP | sed 's/\.npy$//' | sort)
echo "${#OBJS[@]} objects -> GPU ${GPUS[*]}"

run_stream() {  # $1=gpu, $2..=objects
  local gpu=$1; shift
  for o in "$@"; do
    python scripts/validate_grasps_kistar.py --gpu $gpu \
      --mesh_path $MESH --grasp_path $GRASP --result_path $RESULT --object_code "$o" \
      --penetration_threshold $PEN --joints_threshold $JOINTS --spen_threshold $SPEN
  done
}

# round-robin objects to GPUs, run streams in parallel
declare -a B
for i in "${!OBJS[@]}"; do B[$(( i % ${#GPUS[@]} ))]+="${OBJS[$i]} "; done
for g in "${!GPUS[@]}"; do run_stream "${GPUS[$g]}" ${B[$g]} & done
wait

echo "done: validated -> $RESULT  ($(ls $RESULT | wc -l) objects with valid grasps)"
