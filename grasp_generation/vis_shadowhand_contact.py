"""
Visualize shadowhand contact candidates (DexGrasp_KIST mjcf) on the hand mesh,
without pytorch3d. Replicates hand_model.py's mjcf mesh build via pk chain, then
plots lightblue mesh + red contact candidates. Run in the DexGrasp conda env.

  python vis_shadowhand_contact.py --out /tmp/shadowhand_contact.png
"""
import argparse
import json
import os

import numpy as np
import torch
import trimesh as tm
import plotly.graph_objects as go
import pytorch_kinematics as pk

MJCF = "mjcf/shadow_hand_wrist_free.xml"
MESH = "mjcf/meshes"
CONTACT = "mjcf/contact_points.json"


def build_link_meshes(chain, device="cpu"):
    """Traverse chain; return {link_name: (verts(N,3), faces(M,3))} in link-local frame."""
    meshes = {}

    def recurse(body):
        vis = body.link.visuals
        if len(vis) > 0:
            lv, lf, nv = [], [], 0
            for visual in vis:
                scale = None
                if visual.geom_type == "box":
                    lm = tm.load_mesh(os.path.join(MESH, "box.obj"), process=False)
                    lm.vertices *= visual.geom_param.detach().cpu().numpy()
                elif visual.geom_type == "capsule":
                    lm = tm.primitives.Capsule(radius=float(visual.geom_param[0]),
                                               height=float(visual.geom_param[1]) * 2
                                               ).apply_translation((0, 0, -float(visual.geom_param[1])))
                elif visual.geom_type == "mesh":
                    name = visual.geom_param[0].split(":")[1]
                    lm = tm.load_mesh(os.path.join(MESH, name + ".obj"), process=False)
                    if visual.geom_param[1] is not None:
                        scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                else:
                    continue
                v = torch.tensor(lm.vertices, dtype=torch.float, device=device)
                f = torch.tensor(lm.faces, dtype=torch.long, device=device)
                if scale is not None:
                    v = v * scale
                v = visual.offset.to(device).transform_points(v)   # visual local -> link frame
                lv.append(v); lf.append(f + nv); nv += len(v)
            meshes[body.link.name] = (torch.cat(lv, 0), torch.cat(lf, 0))
        for ch in body.children:
            recurse(ch)
    recurse(chain._root)
    return meshes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/shadowhand_contact.png")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--surface", action="store_true",
                    help="show GenDexGrasp-style full-surface sampling (64/link) instead of contact json")
    args = ap.parse_args()

    chain = pk.build_chain_from_mjcf(open(MJCF).read()).to(dtype=torch.float)
    meshes = build_link_meshes(chain)
    n_dof = len(chain.get_joint_parameter_names())
    status = chain.forward_kinematics(torch.zeros(1, n_dof))   # link -> Transform3d

    # pass 1: gather world-frame geometry
    world_meshes = []
    for link, (v, f) in meshes.items():
        if link not in status:
            continue
        Vw = status[link].transform_points(v.unsqueeze(0))[0].numpy()
        world_meshes.append((Vw, f.numpy()))

    if args.surface:
        import trimesh.sample
        cp, nlinks = [], len(world_meshes)
        for Vw, F in world_meshes:
            m = tm.Trimesh(Vw, F, process=False)
            P, _ = trimesh.sample.sample_surface(m, 64)
            cp.append(np.asarray(P))
        cp = np.concatenate(cp, 0)
        pt_color, label = "orange", "full surface (64/link)"
    else:
        contact = json.load(open(CONTACT))
        cp, nlinks = [], 0
        for link, pts in contact.items():
            if not pts or link not in status:
                continue
            nlinks += 1
            P = torch.tensor(pts, dtype=torch.float).reshape(1, -1, 3)
            cp.append(status[link].transform_points(P)[0].numpy())
        cp = np.concatenate(cp, 0)
        pt_color, label = "red", "contact candidates"

    # consistent palm-facing camera + center geometry (fix crop & rotation)
    from vis_cam import palm_view
    allV = np.concatenate([V for V, _ in world_meshes], axis=0)
    C, cam = palm_view(allV, cp)

    data = []
    for Vw, F in world_meshes:
        Vc = Vw - C
        data.append(go.Mesh3d(x=Vc[:, 0], y=Vc[:, 1], z=Vc[:, 2],
                              i=F[:, 0], j=F[:, 1], k=F[:, 2],
                              color="lightblue", opacity=0.45))
    cpc = cp - C
    data.append(go.Scatter3d(x=cpc[:, 0], y=cpc[:, 1], z=cpc[:, 2], mode="markers",
                             marker=dict(size=3, color=pt_color)))

    fig = go.Figure(data=data)
    fig.update_layout(title=f"shadowhand: {label} (n={len(cp)} on {nlinks} links)",
                      scene=dict(aspectmode="data", camera=cam,
                                 xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    ext = os.path.splitext(args.out)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf"):
        fig.write_image(args.out, width=args.width, height=args.height, scale=2)
    else:
        fig.write_html(args.out)
    print(f"shadowhand: {len(cp)} contact pts on {nlinks} links -> {args.out}")


if __name__ == "__main__":
    main()
