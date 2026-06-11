#!/bin/bash
# v4 RE-validation with penetration_threshold=0.01 (vs the 0.001 baseline in
# dataset_v4). Same in every other way (no joints/spen filter) to isolate the
# pen-threshold effect on Isaac-survival yield. Objects split across GPU 4,5.
cd /root/DexGrasp_KIST/grasp_generation
source /root/miniconda3/etc/profile.d/conda.sh && conda activate DexGrasp
PY=/root/miniconda3/envs/DexGrasp/bin/python

mapfile -t OBJS < <(ls ../data/graspdata_v4 | sed 's/\.npy$//' | sort)
echo "총 ${#OBJS[@]} 물체 -> GPU 4,5, pen<0.01 재검증"

run_stream() {  # $1=gpu, $2..=objects
  local gpu=$1; shift
  for obj in "$@"; do
    $PY scripts/validate_grasps_kistar.py --gpu $gpu \
      --mesh_path ../data/meshdata_norm --grasp_path ../data/graspdata_v4 \
      --result_path ../data/dataset_v4_p01 --object_code "$obj" \
      --penetration_threshold 0.01 \
      >> ../data/val_v4_p01_gpu${gpu}.log 2>&1
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
echo "=== v4 pen0.01 재검증 완료 ==="
