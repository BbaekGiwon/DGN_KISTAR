"""
Precompute hand + object meshes for a set of grasps (DexGrasp env).
Saves an .npz bundle that the viser viewer (DGA env) can load without needing
pytorch_kinematics / torchsdf.

Usage:
  CUDA_VISIBLE_DEVICES=6 python tools/precompute_vis.py \
      --grasp_npy ../data/dataset_ncv2/sim/contactdb-mouse_simulated.npy \
      --object_code contactdb-mouse \
      --mesh_path ../data/meshdata_urdf \
      --out ../data/vis_mouse.npz
"""
import os, sys, argparse
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
    rot6d = np.concatenate([R[:, 0], R[:, 1]])
    ja = [qpos[k] for k in JOINTS]
    return np.concatenate([tr, rot6d, ja]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grasp_npy', required=True)
    ap.add_argument('--object_code', required=True)
    ap.add_argument('--mesh_path', default='../data/meshdata_urdf')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=50)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hand = HandModel("kistar/kistar.urdf", "kistar",
                     "kistar/contact_points.json", "kistar/penetration_points.json",
                     0, device)

    grasps = np.load(args.grasp_npy, allow_pickle=True)
    grasps = list(grasps)[:args.max]
    print(f"{len(grasps)} grasps")

    obj_mesh = tm.load(os.path.join(args.mesh_path, args.object_code, "coacd", "decomposed.obj"),
                       force='mesh', process=False)

    def visual_trimesh(i=0):
        """Build the hand mesh from VISUAL (STL) geometry, not the box collision."""
        vs, fs, off = [], [], 0
        for ln in hand.mesh:
            vv = hand.mesh[ln].get('visual_vertices')
            ff = hand.mesh[ln].get('visual_faces')
            if vv is None or len(vv) == 0:
                continue
            v = hand.current_status[ln].transform_points(vv)
            if v.dim() == 3:
                v = v[i]
            v = v @ hand.global_rotation[i].T + hand.global_translation[i]
            vs.append(v.detach().cpu().numpy())
            fs.append(ff.detach().cpu().numpy() + off)
            off += v.shape[0]
        return np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int32)

    hand_v_list, labels, scales = [], [], []
    hand_faces = None
    for g in grasps:
        hp = torch.tensor(hand_pose_from_qpos(g['qpos']), device=device).unsqueeze(0)
        hand.set_parameters(hp)
        v, f = visual_trimesh(0)
        hand_v_list.append(v)
        if hand_faces is None:
            hand_faces = f          # visual topology is identical across grasps
        labels.append(f"idx{g.get('orig_index','?')} Epen={g.get('E_pen',float('nan')):.4f}")
        scales.append(float(g.get('scale', 1.0)))

    np.savez(args.out,
             obj_v=np.asarray(obj_mesh.vertices, dtype=np.float32),
             obj_f=np.asarray(obj_mesh.faces, dtype=np.int32),
             hand_f=hand_faces,
             hand_v=np.stack(hand_v_list),          # (N, V, 3)
             scales=np.asarray(scales, dtype=np.float32),
             labels=np.asarray(labels),
             object_code=args.object_code)
    print(f"saved {args.out}: {len(hand_v_list)} grasps, hand V={hand_v_list[0].shape}, obj V={len(obj_mesh.vertices)}")


if __name__ == '__main__':
    main()
