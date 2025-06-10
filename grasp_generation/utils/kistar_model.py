"""
Last modified date: 2025.06.10
Author: Jialiang Zhang, Ruicheng Wang, Chanyoung Ahn
Description: Class HandModel for KISTAR (URDF)
"""


import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import torch
import transforms3d
import trimesh as tm
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import pytorch_kinematics as pk
from urdf_parser_py.urdf import Robot, Box, Sphere
import pytorch3d.structures
import pytorch3d.ops
import plotly.graph_objects as go
from torchsdf import index_vertices_by_faces, compute_sdf


class HandModel:
    def __init__(
        self,
        urdf_path,
        mesh_path,
        contact_points_path,
        penetration_points_path,
        n_surface_points=0,
        device="cpu",
    ):
        """
        Create a Hand Model for a URDF robot

        Parameters
        ----------
        urdf_path: str
            path to urdf file
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of hand, use fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device

        # load articulation

        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float, device=device)
        self.robot = Robot.from_xml_file(urdf_path)

        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # load contact points and penetration points

        contact_points = (
            json.load(open(contact_points_path, "r"))
            if contact_points_path is not None
            else None
        )
        penetration_points = (
            json.load(open(penetration_points_path, "r"))
            if penetration_points_path is not None
            else None
        )

        self.mesh = {}
        areas = {}
        for link in self.robot.links:
            if link.visual is None or link.collision is None:
                continue
            self.mesh[link.name] = {}
            # load collision mesh
            collision = link.collision
            if type(collision.geometry) == Sphere:
                link_mesh = tm.primitives.Sphere(radius=collision.geometry.radius)
                self.mesh[link.name]['radius'] = collision.geometry.radius
            elif type(collision.geometry) == Box:
                link_mesh = tm.primitives.Box(extents=collision.geometry.size)
                # link_mesh = tm.load_mesh(os.path.join(os.path.dirname(urdf_path), 'meshes', 'box.obj'), process=False)
                # link_mesh.vertices *= np.array(collision.geometry.size) / 2
            else: 
                filename = collision.geometry.filename
                # link_mesh = tm.load_mesh(filename, process=False)
                link_mesh = tm.load_mesh(os.path.join(mesh_path, filename), process=False)

            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if hasattr(collision.geometry, 'scale') and collision.geometry.scale is None:
                collision.geometry.scale = [1, 1, 1]
            scale = torch.tensor(getattr(collision.geometry, 'scale', [1, 1, 1]), dtype=torch.float, device=device)
            translation = torch.tensor(getattr(collision.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device)
            rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(collision.origin, 'rpy', [0, 0, 0])),
                                    dtype=torch.float, device=device)
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation

            self.mesh[link.name].update({
                'vertices': vertices,
                'faces': faces,
            })
            if 'radius' not in self.mesh[link.name]:
                self.mesh[link.name]['face_verts'] = index_vertices_by_faces(vertices, faces)
            areas[link.name] = tm.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy()).area.item()

            # load visual mesh
            visual = link.visual
            filename = visual.geometry.filename
            link_mesh = tm.load_mesh(os.path.join(mesh_path, filename), process=False)

            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if hasattr(visual.geometry, 'scale') and visual.geometry.scale is None:
                visual.geometry.scale = [1, 1, 1]
            scale = torch.tensor(getattr(visual.geometry, 'scale', [1, 1, 1]), dtype=torch.float, device=device)
            translation = torch.tensor(getattr(visual.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device)
            rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(visual.origin, 'rpy', [0, 0, 0])),
                                    dtype=torch.float, device=device)
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update({
                'visual_vertices': vertices,
                'visual_faces': faces,
            })
            # load contact candidates and penetration keypoints
            contact_candidates = torch.tensor(contact_points[link.name], dtype=torch.float32, device=device).reshape(-1, 3)
            penetration_keypoints = torch.tensor(penetration_points[link.name], dtype=torch.float32, device=device).reshape(-1, 3) if penetration_points is not None else None
            
            self.mesh[link.name].update({
                'contact_candidates': contact_candidates,
                'penetration_keypoints': penetration_keypoints,    
            })

        self.joints_lower = torch.tensor([joint.limit.lower for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)
        self.joints_upper = torch.tensor([joint.limit.upper for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)
        
        # sample surface points
        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            # self.mesh[link_name]['surface_points'] = surface_points
            
            if link_name not in {"thumb_basemotor", "index_basemotor", "middle_basemotor", "ring_basemotor"}:
                self.mesh[link_name]['surface_points'] = surface_points
            else:
                self.mesh[link_name]['surface_points'] = torch.zeros((0, 3), dtype=torch.float, device=device)


        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))
        self.surface_points_link_indices = torch.cat([self.link_name_to_link_index[link_name] * torch.ones(self.mesh[link_name]['surface_points'].shape[0], dtype=torch.long, device=device) for link_name in self.mesh])
        
        self.contact_candidates = [self.mesh[link_name]['contact_candidates'] for link_name in self.mesh]
        self.global_index_to_link_index = sum([[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)], [])
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, dtype=torch.long, device=device)
        self.n_contact_candidates = self.contact_candidates.shape[0]
        
        # build collision mask
        self.adjacency_mask = torch.zeros([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=device)
        for joint in self.robot.joints:
            parent_id = self.link_name_to_link_index[joint.parent]
            child_id = self.link_name_to_link_index[joint.child]
            self.adjacency_mask[parent_id, child_id] = True
            self.adjacency_mask[child_id, parent_id] = True
        # self.adjacency_mask[self.link_name_to_link_index['base_link'], self.link_name_to_link_index['link_13.0']] = True
        # self.adjacency_mask[self.link_name_to_link_index['link_13.0'], self.link_name_to_link_index['base_link']] = True

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

        # set joint limits
        self.joints_lower = torch.tensor(
            [
                joint.limit.lower
                for joint in self.robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )
        self.joints_upper = torch.tensor(
            [
                joint.limit.upper
                for joint in self.robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )

        # sample surface points
        total_area = sum(areas.values())
        num_samples = dict(
            [
                (link_name, int(areas[link_name] / total_area * n_surface_points))
                for link_name in self.mesh
            ]
        )
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(
            num_samples.values()
        )
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=device
                ).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(
                self.mesh[link_name]["vertices"].unsqueeze(0),
                self.mesh[link_name]["faces"].unsqueeze(0),
            )
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=100 * num_samples[link_name]
            )
            surface_points = pytorch3d.ops.sample_farthest_points(
                dense_point_cloud, K=num_samples[link_name]
            )[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]["surface_points"] = surface_points

        # indexing
        self.link_name_to_link_index = dict(
            zip([link_name for link_name in self.mesh], range(len(self.mesh)))
        )

        self.contact_candidates = [
            self.mesh[link_name]["contact_candidates"] for link_name in self.mesh
        ]
        self.global_index_to_link_index = sum(
            [
                [i] * len(contact_candidates)
                for i, contact_candidates in enumerate(self.contact_candidates)
            ],
            [],
        )
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=device
        )
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = [
            self.mesh[link_name]["penetration_keypoints"] for link_name in self.mesh
        ]
        self.global_index_to_link_index_penetration = sum(
            [
                [i] * len(penetration_keypoints)
                for i, penetration_keypoints in enumerate(self.penetration_keypoints)
            ],
            [],
        )
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(
            self.global_index_to_link_index_penetration, dtype=torch.long, device=device
        )
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9]
        )
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(
                batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device
            )
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = (
                    self.current_status[link_name]
                    .get_matrix()
                    .unsqueeze(1)
                    .expand(batch_size, n_contact, 4, 4)
                )
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat(
                [
                    self.contact_points,
                    torch.ones(
                        batch_size, n_contact, 1, dtype=torch.float, device=self.device
                    ),
                ],
                dim=2,
            )
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[
                :, :, :3, 0
            ]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(
                1, 2
            ) + self.global_translation.unsqueeze(1)

    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes

        Interiors are positive, exteriors are negative

        Use analytical method and our modified Kaolin package

        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Consider each link seperately:
        #   First, transform x into each link's local reference frame using inversed fk, which gives us x_local
        #   Next, calculate point-to-mesh distances in each link's frame, this gives dis_local
        #   Finally, the maximum over all links is the final distance from one point to the entire ariticulation
        # In particular, the collision mesh of ShadowHand is only composed of Capsules and Boxes
        # We use analytical method to calculate Capsule sdf, and use our modified Kaolin package for other meshes
        # This practice speeds up the reverse penetration calculation
        # Note that we use a chamfer box instead of a primitive box to get more accurate signs
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            if link_name in [
                "robot0:forearm",
                "robot0:wrist_child",
                "robot0:ffknuckle_child",
                "robot0:mfknuckle_child",
                "robot0:rfknuckle_child",
                "robot0:lfknuckle_child",
                "robot0:thbase_child",
                "robot0:thhub_child",
            ]:
                continue
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if "geom_param" not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]["face_verts"]
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                height = self.mesh[link_name]["geom_param"][1] * 2
                radius = self.mesh[link_name]["geom_param"][0]
                nearest_point = x_local.detach().clone()
                nearest_point[:, :2] = 0
                nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
                dis_local = radius - (x_local - nearest_point).norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self):
        """
        Calculate self penetration energy

        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(
            batch_size, 1
        )
        transforms = torch.zeros(
            batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device
        )
        for link_name in self.mesh:
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = (
                self.current_status[link_name]
                .get_matrix()
                .unsqueeze(1)
                .expand(batch_size, self.n_keypoints, 4, 4)
            )
            transforms[mask] = cur[mask]
        points = torch.cat(
            [
                points,
                torch.ones(
                    batch_size,
                    self.n_keypoints,
                    1,
                    dtype=torch.float,
                    device=self.device,
                ),
            ],
            dim=2,
        )
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1, 2))

    def get_surface_points(self):
        """
        Get surface points

        Returns
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            if link_name not in {"thumb_basemotor", "index_basemotor", "middle_basemotor", "ring_basemotor"}:
                n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
                points.append(
                    self.current_status[link_name].transform_points(
                        self.mesh[link_name]["surface_points"]
                    )
                )
                if 1 < batch_size != points[-1].shape[0]:
                    points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_contact_candidates(self):
        """
        Get all contact candidates

        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["contact_candidates"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["contact_candidates"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_penetraion_keypoints(self):
        """
        Get penetration keypoints

        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["penetration_keypoints"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["penetration_keypoints"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_plotly_data(
        self, i, opacity=0.5, color="lightblue", with_contact_points=False, pose=None
    ):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]["vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                )
            )
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                )
            )
        return data

    def get_trimesh_data(self, i):
        """
        Get full mesh
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        data = tm.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += tm.Trimesh(vertices=v, faces=f)
        return data
