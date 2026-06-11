#!/bin/bash
# v4 validation, objects split across GPU 4 and 5 in parallel.
# Each object validated by validate_grasps_kistar.py (penetration prefilter + Isaac).
cd /root/DexGrasp_KIST/grasp_generation
source /root/miniconda3/etc/profile.d/conda.sh && conda activate DexGrasp
PY=/root/miniconda3/envs/DexGrasp/bin/python

mapfile -t OBJS < <(ls ../data/graspdata_v4 | sed 's/\.npy$//' | sort)
echo "총 ${#OBJS[@]} 물체 → GPU 4,5 분배"

run_stream() {  # $1=gpu, $2..=objects
  local gpu=$1; shift
  for obj in "$@"; do
    $PY scripts/validate_grasps_kistar.py --gpu $gpu \
      --mesh_path ../data/meshdata_norm --grasp_path ../data/graspdata_v4 \
      --result_path ../data/dataset_v4 --object_code "$obj" \
      >> ../data/val_v4_gpu${gpu}.log 2>&1
    echo "[gpu$gpu] done: $obj"
  done
}

g4=(); g5=()
for i in "${!OBJS[@]}"; do
  if (( i % 2 == 0 )); then g4+=("${OBJS[$i]}"); else g5+=("${OBJS[$i]}"); fi
done
run_stream 4 "${g4[@]}" &
run_stream 5 "${g5[@]}" &
wait
echo "=== v4 검증 전체 완료 ==="
