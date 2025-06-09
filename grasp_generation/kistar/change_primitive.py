import os
import trimesh
import xml.etree.ElementTree as ET
import numpy as np


def guess_primitive_type(mesh: trimesh.Trimesh) -> str:
    """
    Heuristically determine the best-fit primitive shape based on mesh aspect ratio and symmetry.
    Returns one of: "sphere", "capsule", "box"
    """
    extents = mesh.bounding_box.extents
    ratio = np.sort(extents) / max(extents)

    if np.allclose(ratio, [1.0, 1.0, 1.0], atol=0.15):
        return "sphere"
    elif ratio[0] < 0.5 and ratio[1] < 0.5 and ratio[2] > 1.5:
        return "capsule"
    else:
        return "box"


def convert_to_primitive_auto(stl_path, obj_path):
    mesh = trimesh.load(stl_path, force='mesh')
    prim_type = guess_primitive_type(mesh)
    centroid = mesh.centroid

    if prim_type == 'sphere':
        radius = np.min(mesh.bounding_box.extents) / 2
        primitive = trimesh.primitives.Sphere(radius=radius)
    elif prim_type == 'capsule':
        extents = mesh.bounding_box.extents
        radius = min(extents[:2]) / 2
        height = extents[2]
        primitive = trimesh.primitives.Capsule(radius=radius, height=height)
    else:
        primitive = mesh.bounding_box.to_mesh()

    primitive.apply_translation(centroid)
    primitive.export(obj_path)
    return prim_type


def update_urdf_with_auto_primitives(urdf_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.findall(".//collision//geometry//mesh"):
        filename = mesh.get("filename")
        if filename and filename.lower().endswith(".stl"):
            basename = os.path.splitext(os.path.basename(filename))[0]
            stl_path = os.path.join(input_dir, basename + ".obj")
            obj_path = os.path.join(output_dir, basename + ".obj")

            try:
                prim_type = convert_to_primitive_auto(stl_path, obj_path)
                rel_path = os.path.relpath(obj_path, os.path.dirname(urdf_path)).replace("\\", "/")
                mesh.set("filename", rel_path)
                print(f"✓ Updated: {filename} → {rel_path} ({prim_type})")
            except Exception as e:
                print(f"⚠️ Failed: {filename} — {e}")

    new_urdf_path = urdf_path.replace(".urdf", f"_primitive_auto.urdf")
    tree.write(new_urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"\n✅ URDF updated and saved to: {new_urdf_path}")


# Parameters to run
urdf_path = "./kistar_basic.urdf"
input_dir = "./coll_meshes/kistar"
output_dir = "./prim_meshes/kistar"

# Run the auto-conversion
update_urdf_with_auto_primitives(urdf_path, input_dir, output_dir)
