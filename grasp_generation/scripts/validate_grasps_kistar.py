"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator

python scripts/validate_grasps.py --gpu 0 --object_code sem-Bottle-437678d4bc6be981c8724d5673a063a6

"""

import os
import sys

sys.path.append(os.path.realpath('.'))

from utils.isaac_validator_kistar import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.kistar_model import HandModel
from utils.object_model import ObjectModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_batch', default=100, type=int)
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    parser.add_argument('--grasp_path', default="../data/graspdata", type=str)
    parser.add_argument('--result_path', default="../data/dataset", type=str)
    parser.add_argument('--object_code',
                        default="mujoco-Ecoforms_Plant_Plate_S11Turquoise",
                        type=str)
    # if index is received, then the debug mode is on
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    # paper (original DexGraspNet) validation defaults
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)
    # extra prefilters using the values stored at generation time.
    # default -1 = DISABLED -> exactly preserves old behavior (current v4 run).
    # next validations: pass e.g. --spen_threshold 0.005 --joints_threshold 1e-4
    parser.add_argument('--spen_threshold', default=-1.0, type=float,
                        help='keep grasps with E_spen < this (self-penetration). <0 disables.')
    parser.add_argument('--joints_threshold', default=-1.0, type=float,
                        help='keep grasps with E_joints < this (joint-limit violation). <0 disables.')

    args = parser.parse_args()

    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
        'thumb_joint_0', 'thumb_joint_1', 'thumb_joint_2', 'thumb_joint_3',
        'index_joint_0', 'index_joint_1', 'index_joint_2', 'index_joint_3',
        'middle_joint_0', 'middle_joint_1', 'middle_joint_2', 'middle_joint_3',
        'ring_joint_0', 'ring_joint_1', 'ring_joint_2', 'ring_joint_3',
    ]

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.makedirs(args.result_path, exist_ok=True)

    if not args.no_force:
        device = torch.device(
            f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        data_dict = np.load(os.path.join(
            args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
        batch_size = data_dict.shape[0]
        hand_state = []
        scale_tensor = []
        # from IPython import embed; embed(); exit();
        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            rot = np.array(transforms3d.euler.euler2mat(
                *[qpos[name] for name in rot_names]))
            rot = rot[:, :2].T.ravel().tolist()
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [
                qpos[name] for name in joint_names], dtype=torch.float, device=device)
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)
        # print(scale_tensor.dtype)
        hand_model = HandModel(
            urdf_path="kistar/kistar.urdf",
            mesh_path="kistar",
            contact_points_path="kistar/contact_points.json",
            penetration_points_path="kistar/penetration_points.json",
            n_surface_points=2000,
            device=device,
        )
        hand_model.set_parameters(hand_state)
        # object model
        object_model = ObjectModel(
            data_root_path=args.mesh_path,
            batch_size_each=batch_size,
            num_samples=0,
            device=device
        )
        object_model.initialize(args.object_code)
        object_model.object_scale_tensor = scale_tensor

        # calculate contact points and contact normals
        contact_points_hand = torch.zeros((batch_size, 19, 3)).to(device)
        contact_normals = torch.zeros((batch_size, 19, 3)).to(device)

        for i, link_name in enumerate(hand_model.mesh):
            if len(hand_model.mesh[link_name]['surface_points']) == 0:
                continue
            surface_points = hand_model.current_status[link_name].transform_points(
                hand_model.mesh[link_name]['surface_points']).expand(batch_size, -1, 3)
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2) + hand_model.global_translation.unsqueeze(1)
            distances, normals = object_model.cal_distance(
                surface_points)
            nearest_point_index = distances.argmax(dim=1)
            nearest_distances = torch.gather(
                distances, 1, nearest_point_index.unsqueeze(1))
            nearest_points_hand = torch.gather(
                surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            nearest_normals = torch.gather(
                normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            admited = -nearest_distances < args.thres_cont
            admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
            contact_points_hand[:, i:i+1, :] = torch.where(
                admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])
            contact_normals[:, i:i+1, :] = torch.where(
                admited, nearest_normals, contact_normals[:, i:i+1, :])

        target_points = contact_points_hand + contact_normals * args.dis_move
        loss = (target_points.detach().clone() -
                contact_points_hand).square().sum()
        loss.backward()
        with torch.no_grad():
            hand_state[:, 9:] += hand_state.grad[:, 9:] * args.grad_move
            hand_state.grad.zero_()

    sim = IsaacValidator(gpu=args.gpu, mode="gui" if args.index is not None else None)
    data_dict = np.load(os.path.join(
        args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
    
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    E_pen_array = []
    E_spen_array = []
    E_joints_array = []
    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']
        rot = [qpos[name] for name in rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        translations.append(np.array([qpos[name]
                            for name in translation_names]))
        hand_poses.append(np.array([qpos[name] for name in joint_names]))
        scale_array.append(scale)
        E_pen_array.append(data_dict[i]["E_pen"])
        E_spen_array.append(data_dict[i].get("E_spen", 0.0))
        E_joints_array.append(data_dict[i].get("E_joints", 0.0))
    E_pen_array = np.array(E_pen_array)
    E_spen_array = np.array(E_spen_array)
    E_joints_array = np.array(E_joints_array)
    if not args.no_force:
        hand_poses = hand_state[:, 9:]

    if (args.index is not None):
        sim.set_asset("kistar", "kistar.urdf",
                       os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
        index = args.index
        # print(hand_poses[index])
        # print(hand_poses[index].shape)
        # from IPython import embed; embed(); exit();
        sim.add_env_single(rotations[index], translations[index], hand_poses[index],
                           scale_array[index], 0)
        result = sim.run_sim()
        print(result)
    else:
        # The penetration check uses the precomputed E_pen (free), while the
        # IsaacGym sim is the expensive part. So filter by penetration FIRST and
        # only simulate the survivors -> identical `valid`, much faster.
        estimated = E_pen_array < args.penetration_threshold
        # extra prefilters (disabled when threshold < 0): drop grasps that
        # violate joint limits or self-penetrate BEFORE the expensive Isaac sim.
        # joint limit first (hard constraint: Isaac clamps DOF), then self-pen.
        if args.joints_threshold >= 0:
            keep = E_joints_array < args.joints_threshold
            print(f'  joints filter (E_joints<{args.joints_threshold}): '
                  f'{(estimated & keep).sum()}/{estimated.sum()} survive', flush=True)
            estimated = estimated & keep
        if args.spen_threshold >= 0:
            keep = E_spen_array < args.spen_threshold
            print(f'  spen filter (E_spen<{args.spen_threshold}): '
                  f'{(estimated & keep).sum()}/{estimated.sum()} survive', flush=True)
            estimated = estimated & keep
        sim_indices = np.where(estimated)[0]
        extra = "" if (args.joints_threshold < 0 and args.spen_threshold < 0) else " & joints & spen"
        print(f'estimated (E_pen<{args.penetration_threshold}{extra}): '
              f'{estimated.sum()}/{batch_size} -> Isaac on these only', flush=True)

        simulated = np.zeros(batch_size, dtype=np.bool8)
        result = []
        offset = 0
        while offset < len(sim_indices):
            batch_idx = sim_indices[offset: offset + args.val_batch]
            sim.set_asset("kistar", "kistar.urdf",
                          os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
            for index in batch_idx:
                sim.add_env(rotations[index], translations[index], hand_poses[index],
                            scale_array[index])
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            offset += len(batch_idx)
        for j, index in enumerate(sim_indices):
            simulated[index] = np.array(sum(result[j * 6:(j + 1) * 6]) == 6)

        valid = simulated * estimated  # == simulated, since we only sim estimated
        print(
            f'estimated: {estimated.sum().item()}/{batch_size}, '
            f'simulated: {simulated.sum().item()}/{batch_size}, '
            f'valid: {valid.sum().item()}/{batch_size}')
        result_list = []
        for i in range(batch_size):
            if (valid[i]):
                result_list.append({"qpos": data_dict[i]["qpos"],
                                    "scale": data_dict[i]["scale"]})
        np.save(os.path.join(args.result_path, args.object_code + '.npy'),
                result_list, allow_pickle=True)
        print(f'saved {len(result_list)} valid grasps to {args.object_code}.npy')
    # sim.destroy()

    if args.index is not None:
        print("Viewer running. Press Ctrl+C to exit.")
        try:
            while True:
                sim.run_sim() # 또는 time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")