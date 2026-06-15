#!/bin/bash
# More v5 grasps: same v5 config (new face-aligned spen, 1-sigma-inside init,
# w_spen=20), across GPUs 2,4,5,6,7, looping over seeds for fresh grasps.
# Each seed -> its own result dir (merge the valid sets later).
cd /root/DexGrasp_KIST/grasp_generation
source /root/miniconda3/etc/profile.d/conda.sh && conda activate DexGrasp
export CUDA_VISIBLE_DEVICES=2,4,5,6,7
export OBJ_SCALE=0.03,0.045,0.06,0.075

for seed in 790 791 792 793 794 795; do
  echo "===== seed $seed start ====="
  python scripts/generate_grasps_kistar.py --all --seed $seed \
    --data_root_path ../data/meshdata_norm --result_path ../data/graspdata_v5_s$seed \
    --batch_size_each 1440 --max_total_batch_size 7200 --n_iter 6000 \
    --temperature_decay 0.98 --switch_possibility 0.1 \
    --w_pen 100 --w_spen 20 --w_joints 20 \
    > ../data/gen_v5_s$seed.log 2>&1
  echo "===== seed $seed done ($(ls ../data/graspdata_v5_s$seed 2>/dev/null|wc -l)/78) ====="
done
echo "===== ALL SEEDS DONE ====="
