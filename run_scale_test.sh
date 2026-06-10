#!/bin/bash
# Scale-sensitivity test: regenerate ycb-plum at object scale 1.5x and 2.0x
# (default dataset stays at 1.0x). Then we validate to see if valid grasps rise.
cd /root/DexGrasp_KIST/grasp_generation
PY=/root/miniconda3/envs/DexGrasp/bin/python
mkdir -p ../data/graspdata_scale

gen() { local gpu=$1 sc=$2; \
  OBJ_SCALE=$sc CUDA_VISIBLE_DEVICES=$gpu $PY scripts/generate_grasps_kistar.py \
    --object_code_list ycb-plum --data_root_path ../data/meshdata_urdf \
    --result_path ../data/graspdata_scale/s${sc} \
    --batch_size_each 600 --max_total_batch_size 600 \
    --n_iter 6000 --temperature_decay 0.98 --switch_possibility 0.1 --overwrite \
    > ../data/gen_scale_${sc}.log 2>&1; echo "[gen] scale $sc done"; }

echo "=== scale generation start: $(date) ==="
gen 6 1.5 &
gen 7 2.0 &
wait
echo "=== scale generation done: $(date) ==="

echo "=== validation ==="
mkdir -p ../data/dataset_scale
for sc in 1.5 2.0; do
  CUDA_VISIBLE_DEVICES=6 $PY scripts/validate_grasps_kistar.py \
    --gpu 6 --mesh_path ../data/meshdata_urdf --grasp_path ../data/graspdata_scale/s${sc} \
    --result_path ../data/dataset_scale/s${sc} --object_code ycb-plum > ../data/val_scale_${sc}.log 2>&1
  echo "[scale $sc] $(grep -iE 'estimated|simulated|valid:' ../data/val_scale_${sc}.log | tail -1)"
done
echo "=== ALL DONE: $(date) ==="
