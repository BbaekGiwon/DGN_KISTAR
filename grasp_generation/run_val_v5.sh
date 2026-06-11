#!/bin/bash
# v5 Isaac validation, objects split across GPUs 2,4,5,6,7 for speed.
# Filters: pen<0.01, joints<0.001, spen<0.005 (locked thresholds; v5 stored
# E_spen already uses the new face-aligned keypoints).
cd /root/DexGrasp_KIST/grasp_generation
source /root/miniconda3/etc/profile.d/conda.sh && conda activate DexGrasp
PY=/root/miniconda3/envs/DexGrasp/bin/python
GPUS=(2 4 5 6 7)

mapfile -t OBJS < <(ls ../data/graspdata_v5 | sed 's/\.npy$//' | sort)
echo "총 ${#OBJS[@]} 물체 -> GPU ${GPUS[*]}"

run_stream() {  # $1=gpu, $2..=objects
  local gpu=$1; shift
  for obj in "$@"; do
    $PY scripts/validate_grasps_kistar.py --gpu $gpu \
      --mesh_path ../data/meshdata_norm --grasp_path ../data/graspdata_v5 \
      --result_path ../data/dataset_v5 --object_code "$obj" \
      --penetration_threshold 0.01 --joints_threshold 0.001 --spen_threshold 0.005 \
      >> ../data/val_v5_gpu${gpu}.log 2>&1
    echo "[gpu$gpu] done: $obj"
  done
}

# round-robin assign objects to the 5 GPUs
declare -a buckets
for i in "${!OBJS[@]}"; do
  g=$(( i % ${#GPUS[@]} ))
  buckets[$g]+="${OBJS[$i]} "
done
for g in "${!GPUS[@]}"; do
  run_stream "${GPUS[$g]}" ${buckets[$g]} &
done
wait
echo "=== v5 검증 전체 완료 ==="
