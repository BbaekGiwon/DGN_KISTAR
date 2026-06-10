"""
Batch-precompute visual hand + object meshes for the all-3-passing grasps of
EVERY object in a graspdata dir. Loads the hand model once. Writes one
vis_<object>.npz per object into --out_dir (skips objects with 0 passing).

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/precompute_all.py \
      --grasp_dir ../data/graspdata_100_50 --mesh_path ../data/meshdata_urdf \
      --out_dir ../data/vis_all --max 30
"""
import os, sys, glob, argparse
sys.path.append(os.path.realpath('.'))
import numpy as np
import torch
import transforms3d
import trimesh as tm
from utils.kistar_model import HandModel

JOINTS = ['thumb_joint_0','thumb_joint_1','thumb_joint_2','thumb_joint_3',
          'index_joint_0','index_joint_1','index_joint_2','index_joint_3',
          'middle_joint_0','middle_joint_1','middle_joint_2','middle_joint_3',
          'ring_joint_0','ring_joint_1','ring_joint_2','ring_joint_3']


def hand_pose_from_qpos(qpos):
    tr = [qpos[k] for k in ['WRJTx', 'WRJTy', 'WRJTz']]
    eu = [qpos[k] for k in ['WRJRx', 'WRJRy', 'WRJRz']]
    R = transforms3d.euler.euler2mat(*eu, axes='sxyz')
    return np.concatenate([tr, R[:, 0], R[:, 1], [qpos[k] for k in JOINTS]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grasp_dir', default='../data/graspdata_100_50')
    ap.add_argument('--mesh_path', default='../data/meshdata_urdf')
    ap.add_argument('--out_dir', default='../data/vis_all')
    ap.add_argument('--max', type=int, default=30)
    ap.add_argument('--tf', type=float, default=1.0)
    ap.add_argument('--tp', type=float, default=0.02)
    ap.add_argument('--ts', type=float, default=0.02)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hand = HandModel("kistar/kistar.urdf", "kistar",
                     "kistar/contact_points.json", "kistar/penetration_points.json", 0, device)

    def visual_mesh(i=0):
        vs, fs, off = [], [], 0
        for ln in hand.mesh:
            vv = hand.mesh[ln].get('visual_vertices'); ff = hand.mesh[ln].get('visual_faces')
            if vv is None or len(vv) == 0:
                continue
            v = hand.current_status[ln].transform_points(vv)
            if v.dim() == 3:
                v = v[i]
            v = v @ hand.global_rotation[i].T + hand.global_translation[i]
            vs.append(v.detach().cpu().numpy()); fs.append(ff.detach().cpu().numpy() + off)
            off += v.shape[0]
        return np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int32)

    files = sorted(glob.glob(os.path.join(args.grasp_dir, '*.npy')))
    summary = []
    for f in files:
        obj = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        sel = [g for g in d if g['E_fc'] < args.tf and g['E_pen'] < args.tp and g['E_spen'] < args.ts]
        sel = sel[:args.max]
        if not sel:
            summary.append((obj, 0)); continue
        om = tm.load(os.path.join(args.mesh_path, obj, "coacd", "decomposed.obj"), force='mesh', process=False)
        hand_v, hand_f, labels = [], None, []
        for g in sel:
            hp = torch.tensor(hand_pose_from_qpos(g['qpos']), device=device).unsqueeze(0)
            hand.set_parameters(hp)
            v, ff = visual_mesh(0)
            hand_v.append(v)
            if hand_f is None:
                hand_f = ff
            labels.append(f"Efc={g['E_fc']:.2f} Epen={g['E_pen']:.3f} Espen={g['E_spen']:.3f}")
        np.savez(os.path.join(args.out_dir, f'{obj}.npz'),
                 obj_v=np.asarray(om.vertices, dtype=np.float32),
                 obj_f=np.asarray(om.faces, dtype=np.int32),
                 hand_f=hand_f, hand_v=np.stack(hand_v),
                 labels=np.asarray(labels), object_code=obj)
        summary.append((obj, len(sel)))
        print(f"  {obj:34s} {len(sel)}")
    print(f"\ndone: {sum(1 for _,n in summary if n>0)} objects with grasps, "
          f"{sum(n for _,n in summary)} total grasps -> {args.out_dir}")


if __name__ == '__main__':
    main()
