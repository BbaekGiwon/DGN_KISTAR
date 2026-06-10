"""
Collect ALL valid (Isaac-passing) grasps across objects into one flat bundle for
visualization. Each entry carries its own hand mesh + object mesh + label.

Usage (DexGrasp env, GPU):
  CUDA_VISIBLE_DEVICES=2 python tools/precompute_valid.py \
      --dataset_dir ../data/dataset_v3 --mesh_path ../data/meshdata_urdf \
      --out ../data/vis_valid_v3.npy
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


def hand_pose(qpos):
    tr = [qpos[k] for k in ['WRJTx', 'WRJTy', 'WRJTz']]
    R = transforms3d.euler.euler2mat(*[qpos[k] for k in ['WRJRx','WRJRy','WRJRz']], axes='sxyz')
    return np.concatenate([tr, R[:, 0], R[:, 1], [qpos[k] for k in JOINTS]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', default='../data/dataset_v3')
    ap.add_argument('--mesh_path', default='../data/meshdata_urdf')
    ap.add_argument('--out', default='../data/vis_valid_v3.npy')
    args = ap.parse_args()

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

    obj_cache = {}
    entries = []
    for f in sorted(glob.glob(os.path.join(args.dataset_dir, '*.npy'))):
        if f.endswith('_simulated.npy'):
            continue
        obj = os.path.basename(f)[:-4]
        grasps = np.load(f, allow_pickle=True)
        if len(grasps) == 0:
            continue
        if obj not in obj_cache:
            m = tm.load(os.path.join(args.mesh_path, obj, "coacd", "decomposed.obj"),
                        force='mesh', process=False)
            obj_cache[obj] = (np.asarray(m.vertices, dtype=np.float32),
                              np.asarray(m.faces, dtype=np.int32))
        ov, of = obj_cache[obj]
        for g in grasps:
            hp = torch.tensor(hand_pose(g['qpos']), device=device).unsqueeze(0)
            hand.set_parameters(hp)
            hv, hf = visual_mesh(0)
            entries.append({'object': obj, 'hand_v': hv, 'hand_f': hf,
                            'obj_v': ov * float(g.get('scale', 1.0)), 'obj_f': of})
        print(f"  {obj}: +{len(grasps)}")
    np.save(args.out, entries, allow_pickle=True)
    print(f"\nsaved {len(entries)} valid grasps from {len(obj_cache)} objects -> {args.out}")


if __name__ == '__main__':
    main()
