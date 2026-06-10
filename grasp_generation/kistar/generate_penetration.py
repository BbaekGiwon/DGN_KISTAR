"""
Generate self-penetration keypoints that COVER each link's volume, instead of a
single center point. A keypoint pair within 2cm is penalized (E_spen), so one
center point (~1cm effective radius) leaves the finger sides/ends uncovered ->
inter-finger / tip collisions are missed. Here we lay a z*y grid on each finger
link box so the 1cm spheres overlap to cover the box. Same-link / adjacent-link
pairs are masked out in self_penetration, so denser points don't add a baseline.

Run from the kistar/ dir:  python generate_penetration.py
"""
import xml.etree.ElementTree as ET
import numpy as np
import json
import os
import trimesh as tm


def box_grid(size, origin, spacing=0.018):
    """Grid of points on a box (centered at origin), covering it with ~spacing.
    Points sit at the box center in x; spread over y (width) and z (length).
    ceil so the 1cm spheres (2cm threshold) overlap to cover each axis -> the
    finger SIDES (y) get >=2 points, closing the inter-finger collision gap."""
    import math
    sx, sy, sz = size
    ox, oy, oz = origin
    R = 0.01                              # keypoint radius = threshold(0.02)/2
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    # --- x-y cross-section pattern: circle passes through the cross-section
    #     corners. The FULL radius budget goes to the cross-section, so the y
    #     offset stays small (dy) and adjacent fingers don't trigger a baseline.
    if hx < R:
        dy = hy - math.sqrt(R * R - hx * hx)        # thin: 2 points on y-axis
        xy = [(0.0, -dy), (0.0, dy)]
    else:
        ax = max(hx - R / math.sqrt(2), 0.0)        # square base: 2x2 corners
        ay = max(hy - R / math.sqrt(2), 0.0)
        xy = [(sxx * ax, syy * ay) for sxx in (-1, 1) for syy in (-1, 1)]

    # --- z: simply 1cm (=R) inside each end -> 2 evenly-offset levels. Keeps the
    #     y pair (width) but spreads the z points so a collision isn't counted by
    #     many clustered points. Short links collapse to a single center level.
    if sz >= 2 * R:
        zs = [oz - hz + R, oz + hz - R]
    else:
        zs = [oz]
    return [[float(ox + x), float(oy + y), float(z)] for z in zs for (x, y) in xy]


def main():
    tree = ET.parse("kistar.urdf")
    root = tree.getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name", "")
        col = link.find("collision")
        pts = []
        if col is not None:
            geom = col.find("geometry")
            box = geom.find("box") if geom is not None else None
            mesh = geom.find("mesh") if geom is not None else None
            org = col.find("origin")
            off = [float(s) for s in org.attrib["xyz"].split()] if (org is not None and "xyz" in org.attrib) else [0, 0, 0]
            if box is not None:
                size = [float(s) for s in box.attrib["size"].split()]
                # 1 center point for: the whole thumb (no self-penetration in
                # practice) and every base link_0 (it barely moves with abduction
                # and any inter-finger collision shows up far more on the upper
                # flexing links). The flexing link_1/2 keep full coverage.
                single = name.startswith("thumb") or name.endswith("link_0")
                pts = [list(off)] if single else box_grid(size, off)
            elif mesh is not None:
                # tips are meshes -> single point at the mesh bbox center (+ origin)
                fn = mesh.attrib["filename"]
                m = tm.load(os.path.join(".", fn), force="mesh", process=False)
                c = (m.bounds[0] + m.bounds[1]) / 2
                pts = [[float(c[0] + off[0]), float(c[1] + off[1]), float(c[2] + off[2])]]

        # keep mount / palm / basemotors empty (palm self-collision is prevented
        # by joint limits; ShadowHand/Allegro also use no palm penetration points)
        if name in {"mount", "palm", "thumb_basemotor", "index_basemotor",
                    "middle_basemotor", "ring_basemotor"}:
            out[name] = []
        else:
            out[name] = pts

    with open("penetration_points.json", "w") as f:
        json.dump(out, f, indent=2)
    tot = sum(len(v) for v in out.values())
    print(f"Saved penetration_points.json: {tot} keypoints")
    for k, v in out.items():
        if v:
            print(f"  {k:16s}: {len(v)}")


if __name__ == "__main__":
    main()
