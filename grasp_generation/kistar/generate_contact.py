import xml.etree.ElementTree as ET
import numpy as np
import trimesh as tm
import json
import os


def sample_inner_face(size, origin, n_z=3, n_y=2):
    """Sample contact candidates on the +X (inner/grasping) face of a finger-link
    collision box. Box is centered at the collision <origin>; rpy assumed 0."""
    sx, sy, sz = size
    ox, oy, oz = origin
    xs = ox + sx / 2.0
    ys = oy + np.linspace(-sy / 4.0, sy / 4.0, n_y)
    zs = oz + np.linspace(-sz / 4.0, sz / 4.0, n_z)
    return [[float(xs), float(y), float(z)] for y in ys for z in zs]


def _fps(pts, n):
    pts = np.asarray(pts, dtype=float)
    idx = [0]
    d = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(1, n):
        i = int(d.argmax())
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[idx]


def sample_tip_points(mesh_file, n=5):
    """Sample n contact candidates on the inner-distal pad of a fingertip mesh.
    Inner = +X side (fingers curl toward +X); distal = upper Z. Uses mesh
    vertices (guaranteed on surface) filtered to the inner-distal region + FPS."""
    m = tm.load(mesh_file, force="mesh", process=False)
    V = m.vertices
    xmax = V[:, 0].max()
    zmin, zmax = V[:, 2].min(), V[:, 2].max()
    mask = (V[:, 0] > xmax - 0.004) & (V[:, 2] > zmin + 0.35 * (zmax - zmin))
    cand = V[mask]
    if len(cand) < n:
        cand = V[V[:, 0] > xmax - 0.006]
    return [[float(a) for a in p] for p in _fps(cand, n)]


def parse_urdf_and_generate_json(urdf_path, output_json):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    base = os.path.dirname(os.path.abspath(urdf_path))
    contact_dict = {}

    for link in root.findall("link"):
        name = link.attrib["name"]
        contact_points = []

        collision = link.find("collision")
        if collision is not None:
            geometry = collision.find("geometry")
            box = geometry.find("box") if geometry is not None else None
            if box is not None:
                size = [float(s) for s in box.attrib["size"].split()]
                origin = collision.find("origin")
                off = [float(s) for s in origin.attrib["xyz"].split()] \
                    if (origin is not None and "xyz" in origin.attrib) else [0.0, 0.0, 0.0]
                contact_points = sample_inner_face(size, off)

        if name in {"palm", "mount", "thumb_basemotor", "index_basemotor",
                    "middle_basemotor", "ring_basemotor"} or name.endswith("_link_0"):
            # palm, mount, base motors, and the base (abduction) link carry no contacts
            contact_dict[name] = []
        elif name.endswith("_tip"):
            mesh_file = os.path.join(base, "meshes", "kistar",
                                     "thumb_tip.STL" if name == "thumb_tip" else "finger_tip.STL")
            contact_dict[name] = sample_tip_points(mesh_file, n=5)
        else:
            contact_dict[name] = contact_points

    with open(output_json, "w") as f:
        json.dump(contact_dict, f, indent=2)
    print(f"Saved contact_points.json to {output_json}")


urdf_path = "./kistar.urdf"
output_json = "./contact_points.json"
parse_urdf_and_generate_json(urdf_path, output_json)
