#!/bin/bash
# Autonomous KISTAR grasp pipeline: sweep -> pick best config -> full 78-object run.
# Runs entirely on GPU 6,7. Launched inside tmux so it survives logout.

cd /root/DexGrasp_KIST/grasp_generation
PY=/root/miniconda3/envs/DexGrasp/bin/python
DATA=../data
mkdir -p $DATA/sweep

echo "########## PHASE 1: hyperparameter sweep (contactdb-mouse) ##########"
echo "start: $(date)"
sweep() { local gpu=$1 tag=$2; shift 2; \
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/generate_grasps_kistar.py \
    --object_code_list contactdb-mouse \
    --data_root_path $DATA/meshdata_urdf --result_path $DATA/sweep/$tag \
    --batch_size_each 600 --max_total_batch_size 600 --overwrite "$@" \
    > $DATA/sweep_$tag.log 2>&1; echo "[sweep] $tag finished"; }

# all configs use n_iter 6000 so the winning param string maps directly to the full run
sweep 6 cfg0_baseline  --n_iter 6000 &
sweep 6 cfg1_slowdecay --n_iter 6000 --temperature_decay 0.98 &
sweep 7 cfg2_lowswitch --n_iter 6000 --switch_possibility 0.1 &
sweep 7 cfg3_combo     --n_iter 6000 --temperature_decay 0.98 --switch_possibility 0.1 &
wait
echo "sweep done: $(date)"

echo "########## PHASE 2: select best config by yield ##########"
BEST=$($PY - <<'PYEOF'
import numpy as np, os, sys
# tie-break order: prefer more-exploratory configs first
cfgs=[
 ('cfg3_combo','--n_iter 6000 --temperature_decay 0.98 --switch_possibility 0.1'),
 ('cfg1_slowdecay','--n_iter 6000 --temperature_decay 0.98'),
 ('cfg2_lowswitch','--n_iter 6000 --switch_possibility 0.1'),
 ('cfg0_baseline','--n_iter 6000'),
]
best=None; best_y=-1.0
for tag,params in cfgs:
    f=f'../data/sweep/{tag}/contactdb-mouse.npy'
    if not os.path.exists(f):
        print(f'{tag}: MISSING',file=sys.stderr); continue
    d=np.load(f,allow_pickle=True)
    efc=np.array([g['E_fc'] for g in d]); edis=np.array([g['E_dis'] for g in d]); epen=np.array([g['E_pen'] for g in d])
    y=float(((efc<1.0)&(edis<0.015)&(epen<0.02)).mean())
    print(f'{tag}: yield={y*100:.2f}%  E_fc_med={np.median(efc):.2f} E_dis_med={np.median(edis):.4f}',file=sys.stderr)
    if y>best_y: best_y=y; best=params
print(best if best is not None else '--n_iter 6000')
PYEOF
)
echo "BEST PARAMS: $BEST"

echo "########## PHASE 3: full 78-object generation (GPU 6,7) ##########"
echo "start: $(date)"
CUDA_VISIBLE_DEVICES=6,7 $PY scripts/generate_grasps_kistar.py --all \
  --data_root_path $DATA/meshdata_urdf --result_path $DATA/graspdata_urdf \
  --overwrite $BEST > $DATA/full78.log 2>&1
echo "########## ALL DONE: $(date) ##########"
