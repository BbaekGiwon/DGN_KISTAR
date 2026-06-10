"""
Visualize how each hand's contact candidates (contact_points.json) are laid out
on the hand mesh -- DexGrasp_KIST style (lightblue mesh + red contact markers),
but without pytorch3d: uses pytorch_kinematics FK + trimesh meshes + plotly/kaleido.

Run in the `gendexgrasp` conda env from /root/DexGrasp_KIST/grasp_generation:
  python vis_other_hands_contact.py --hand allegro --out /tmp/allegro_contact.png
  python vis_other_hands_contact.py --hand kistar  --out /tmp/kistar_dgk_contact.png
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
import pytorch_kinematics as pk

HANDS = {
    "allegro": dict(urdf="allegro_hand_description/allegro_hand_description_right.urdf",
                    contact="allegro_hand_description/contact_points.json",
                    mesh_base="allegro_hand_description"),
    "kistar":  dict(urdf="kistar/kistar.urdf",
                    contact="kistar/contact_points.json",
                    mesh_base="grasp_generation_root"),  # urdf filenames are relative to cwd
}


def link_visuals(urdf_path):
    """Parse per-link visual mesh info: name -> list of (filename, xyz, rpy, scale)."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        vlist = []
        for vis in link.findall("visual"):
            g = vis.find("geometry")
            mesh = g.find("mesh") if g is not None else None
            if mesh is None:
                continue
            fn = mesh.attrib["filename"]
            sc = mesh.attrib.get("scale", "1 1 1")
            sc = [float(s) for s in sc.split()]
            o = vis.find("origin")
            xyz = [float(s) for s in o.attrib.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(s) for s in o.attrib.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            vlist.append((fn, xyz, rpy, sc))
        out[link.attrib["name"]] = vlist
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True, choices=list(HANDS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--surface", action="store_true",
                    help="show GenDexGrasp-style full-surface sampling (128/link) instead of contact json")
    args = ap.parse_args()
    cfg = HANDS[args.hand]
    import json

    urdf_path = cfg["urdf"]
    chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float)
    joint_names = chain.get_joint_parameter_names()
    th = torch.zeros(1, len(joint_names))
    ret = chain.forward_kinematics(th)  # dict link -> Transform3d

    visuals = link_visuals(urdf_path)
    mesh_base = "." if cfg["mesh_base"] == "grasp_generation_root" else cfg["mesh_base"]

    # --- pass 1: gather all geometry in world frame ---
    meshes = []         # list of (V, F)
    urdf_dir = os.path.dirname(urdf_path)
    for link, vlist in visuals.items():
        if link not in ret:
            continue
        T = ret[link].get_matrix()[0].numpy()  # (4,4) link frame in world
        for fn, xyz, rpy, sc in vlist:
            fn2 = fn.replace("package://", "")            # strip ROS pkg prefix
            cands = [os.path.join(urdf_dir, fn2), fn2, os.path.join(mesh_base, fn2)]
            mpath = next((c for c in cands if os.path.exists(c)), None)
            if mpath is None:
                print(f"  [warn] mesh missing: {fn2}"); continue
            m = tm.load(mpath, force="mesh", process=False)
            V = np.asarray(m.vertices) * np.array(sc)
            Rv = transforms3d.euler.euler2mat(*rpy)
            V = (Rv @ V.T).T + np.array(xyz)              # visual origin -> link frame
            V = (T[:3, :3] @ V.T).T + T[:3, 3]            # link frame -> world
            meshes.append((V, np.asarray(m.faces)))

    if args.surface:
        # GenDexGrasp-style: sample the full surface of every link mesh (128/link)
        import trimesh.sample
        cp_world, n_links_with_cp = [], len(meshes)
        for V, F in meshes:
            m = tm.Trimesh(V, F, process=False)
            P, _ = trimesh.sample.sample_surface(m, 128)
            cp_world.append(np.asarray(P))
        cp_world = np.concatenate(cp_world, axis=0)
        pt_color, label = "orange", "full surface (128/link)"
    else:
        contact = json.load(open(cfg["contact"]))
        cp_world, n_links_with_cp = [], 0
        for link, pts in contact.items():
            if not pts or link not in ret:
                continue
            n_links_with_cp += 1
            T = ret[link].get_matrix()[0].numpy()
            P = np.asarray(pts, dtype=float)
            cp_world.append((T[:3, :3] @ P.T).T + T[:3, 3])
        cp_world = np.concatenate(cp_world, axis=0)
        pt_color, label = "red", "contact candidates"

    # --- consistent palm-facing camera + center geometry (fix crop & rotation) ---
    from vis_cam import palm_view
    allV = np.concatenate([V for V, _ in meshes], axis=0)
    C, cam = palm_view(allV, cp_world)

    data = []
    for V, F in meshes:
        Vc = V - C
        data.append(go.Mesh3d(x=Vc[:, 0], y=Vc[:, 1], z=Vc[:, 2],
                              i=F[:, 0], j=F[:, 1], k=F[:, 2],
                              color="lightblue", opacity=0.45))
    cpc = cp_world - C
    data.append(go.Scatter3d(x=cpc[:, 0], y=cpc[:, 1], z=cpc[:, 2],
                             mode="markers", marker=dict(size=3, color=pt_color)))

    fig = go.Figure(data=data)
    fig.update_layout(title=f"{args.hand}: {label} (n={len(cp_world)} on {n_links_with_cp} links)",
                      scene=dict(aspectmode="data", camera=cam,
                                 xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    ext = os.path.splitext(args.out)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf"):
        fig.write_image(args.out, width=args.width, height=args.height, scale=2)
    else:
        fig.write_html(args.out)
    print(f"{args.hand}: {len(cp_world)} contact pts on {n_links_with_cp} links -> {args.out}")


if __name__ == "__main__":
    main()
