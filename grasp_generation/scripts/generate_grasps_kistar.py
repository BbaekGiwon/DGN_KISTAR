"""
Last modified date: 2025.06.10
Author: Jialiang Zhang, Ruicheng Wang, Chanyoung Ahn
Description: generate grasps in large-scale, use multiple graphics cards, no logging

python scripts/validate_grasps.py --gpu 0 --index 3 --object_code core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03

"""

import os
import sys

sys.path.append(os.path.realpath('.'))

import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random
import transforms3d
import wandb

from utils.kistar_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')

def generate(args_list):
    args, object_code_list, id, gpu_list = args_list

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare models
    n_objects = len(object_code_list)

    identity = multiprocessing.current_process()._identity
    worker = identity[0] if identity else 1
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[worker - 1]
    device = torch.device('cuda')

    # wandb init (one run per worker)
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=f'{args.wandb_name}_w{worker}' if args.wandb_name else None,
            config=vars(args),
            reinit=True,
        )

    hand_model = HandModel(
        urdf_path="kistar/kistar.urdf",
        mesh_path=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "kistar"),
        contact_points_path="kistar/contact_points.json",
        penetration_points_path="kistar/penetration_points.json",
        n_surface_points=1000, 
        device=device,
    )

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=2000, 
        device=device
    )
    object_model.initialize(object_code_list)

    initialize_convex_hull(hand_model, object_model, args)
    
    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'step_size': args.step_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device
    }
    optimizer = Annealing(hand_model, **optim_config)

    # optimize
    
    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_spen=args.w_spen,
        w_joints=args.w_joints,
    )
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    energy.sum().backward(retain_graph=True)

    # for step in range(1, args.n_iter + 1):
    #     s = optimizer.try_step()
    #     optimizer.zero_grad()
    #     new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    #     new_energy.sum().backward(retain_graph=True)

    #     with torch.no_grad():
    #         accept, t = optimizer.accept_step(energy, new_energy)

    #         energy[accept] = new_energy[accept]
    #         E_dis[accept] = new_E_dis[accept]
    #         E_fc[accept] = new_E_fc[accept]
    #         E_pen[accept] = new_E_pen[accept]
    #         E_spen[accept] = new_E_spen[accept]
    #         E_joints[accept] = new_E_joints[accept]

    n_accept_total = 0
    for step in range(1, args.n_iter + 1):
        s = optimizer.try_step()
        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)
        new_energy.sum().backward(retain_graph=True)
        with torch.no_grad():
            accept, temp = optimizer.accept_step(energy, new_energy)
            n_accept = accept.sum().item()
            n_accept_total += n_accept
            if accept.any():
                energy[accept]   = new_energy[accept]
                E_dis[accept]    = new_E_dis[accept]
                E_fc[accept]     = new_E_fc[accept]
                E_pen[accept]    = new_E_pen[accept]
                E_spen[accept]   = new_E_spen[accept]
                E_joints[accept] = new_E_joints[accept]

        if step % 500 == 0:
            accept_rate = n_accept_total / (500 * energy.shape[0])
            n_accept_total = 0
            print(f"[{step}] energy={energy.mean():.3f}  E_fc={E_fc.mean():.3f}  "
                  f"E_dis={E_dis.mean():.4f}  E_pen={E_pen.mean():.4f}  "
                  f"accept={accept_rate*100:.1f}%  T={temp:.3f}", flush=True)
            if args.wandb:
                wandb.log({
                    'step': step,
                    'energy':   energy.mean().item(),
                    'E_fc':     E_fc.mean().item(),
                    'E_dis':    E_dis.mean().item(),
                    'E_pen':    E_pen.mean().item(),
                    'E_spen':   E_spen.mean().item(),
                    'E_joints': E_joints.mean().item(),
                    'temperature': temp.item(),
                    'accept_rate': accept_rate,
                })


    # save results
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
        'thumb_joint_0',  'thumb_joint_1',  'thumb_joint_2',  'thumb_joint_3',
        'index_joint_0',  'index_joint_1',  'index_joint_2',  'index_joint_3',
        'middle_joint_0', 'middle_joint_1', 'middle_joint_2', 'middle_joint_3',
        'ring_joint_0',   'ring_joint_1',   'ring_joint_2',   'ring_joint_3',
    ]

    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(args.batch_size_each):
            idx = i * args.batch_size_each + j
            scale = object_model.object_scale_tensor[i][j].item()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(zip(joint_names, hand_pose[9:].tolist()))

            rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
            euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
            qpos.update(dict(zip(rot_names, euler)))
            qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            hand_pose = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
            rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
            euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
            qpos_st.update(dict(zip(rot_names, euler)))
            qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            data_list.append(dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                energy=energy[idx].item(),
                E_fc=E_fc[idx].item(),
                E_dis=E_dis[idx].item(),
                E_pen=E_pen[idx].item(),
                E_spen=E_spen[idx].item(),
                E_joints=E_joints[idx].item(),
            ))
        np.save(os.path.join(args.result_path, object_code + '.npy'), data_list, allow_pickle=True)

        # wandb: per-object final summary
        if args.wandb:
            e_fc_arr  = np.array([d['E_fc']  for d in data_list])
            e_dis_arr = np.array([d['E_dis'] for d in data_list])
            e_pen_arr = np.array([d['E_pen'] for d in data_list])
            fc_pass = (e_fc_arr < args.thres_fc) & (e_dis_arr < args.thres_dis) & (e_pen_arr < args.thres_pen)
            wandb.log({
                f'{object_code}/fc_pass_rate': fc_pass.mean(),
                f'{object_code}/fc_pass_n':    fc_pass.sum(),
                f'{object_code}/E_fc_mean':    e_fc_arr.mean(),
                f'{object_code}/E_dis_mean':   e_dis_arr.mean(),
                f'{object_code}/E_pen_mean':   e_pen_arr.mean(),
            })
            print(f'  [{object_code}] FC pass: {fc_pass.sum()}/{len(data_list)} = {fc_pass.mean()*100:.1f}%', flush=True)

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--result_path', default="../data/graspdata", type=str)
    parser.add_argument('--data_root_path', default="../data/meshdata", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    # wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='kistar-grasp-gen', type=str)
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--max_total_batch_size', default=600, type=int)
    parser.add_argument('--batch_size_each', default=600, type=int)
    parser.add_argument('--n_iter', default=6000, type=int)
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--step_size', default=0.005, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=18, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=100.0, type=float)
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_joints', default=1.0, type=float)
    # initialization settings
    parser.add_argument('--jitter_strength', default=0.1, type=float)
    parser.add_argument('--distance_lower', default=0.02, type=float)
    parser.add_argument('--distance_upper', default=0.05, type=float)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.005, type=float)

    args = parser.parse_args()

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f'gpu_list: {gpu_list}')
    import time
    start = time.time()
    # check whether arguments are valid and process arguments

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    if not os.path.exists(args.data_root_path):
        raise ValueError(f'data_root_path {args.data_root_path} doesn\'t exist')
    
    if (args.object_code_list is not None) + args.all != 1:
        raise ValueError('exactly one among \'object_code_list\' \'all\' should be specified')
    
    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = os.listdir(args.data_root_path)
    
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError('object_code_list isn\'t a subset of dirs in data_root_path')
    else:
        object_code_list = object_code_list_all
    
    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, object_code + '.npy')):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(f'batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}')
    
    print(f'n_objects: {len(object_code_list)}')
    
    # generate

    random.seed(args.seed)
    random.shuffle(object_code_list)
    objects_each = args.max_total_batch_size // args.batch_size_each
    object_code_groups = [object_code_list[i: i + objects_each] for i in range(0, len(object_code_list), objects_each)]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    if len(gpu_list) == 1:
        for i, args_tuple in enumerate(process_args):
            print(f"[{i+1}/{len(process_args)}] Processing {args_tuple[1]}...")
            generate(args_tuple)
    else:
        with multiprocessing.Pool(len(gpu_list)) as p:
            it = tqdm(p.imap(generate, process_args), total=len(process_args), desc='generating', maxinterval=1000)
            list(it)

    print(f'Finished in {time.time() - start:.2f} seconds.')