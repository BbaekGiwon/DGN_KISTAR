import os
import trimesh
import xml.etree.ElementTree as ET
import numpy as np


def assign_primitive_by_link_name(link_name: str, mesh: trimesh.Trimesh):
    extents = mesh.bounding_box.extents
    if link_name.endswith("_tip"):
        radius = max(extents) / 2
        return "sphere", {"radius": radius}
    elif "mount" in link_name:
        radius = min(extents[:2]) / 2
        length = extents[2]
        return "cylinder", {"radius": radius, "length": length}
    else:
        return "box", {"size": extents.tolist()}


def find_matching_visual_origin(link_element):
    for visual in link_element.findall("visual"):
        origin_tag = visual.find("origin")
        if origin_tag is not None:
            return origin_tag
    return None


def update_urdf_collision_with_named_primitives(urdf_path, input_dir):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        link_name = link.get("name", "")

        for collision in link.findall("collision"):
            mesh_tag = collision.find("geometry/mesh")
            if mesh_tag is None:
                continue

            filename = mesh_tag.get("filename")
            if not filename or not filename.lower().endswith(".stl"):
                continue

            basename = os.path.splitext(os.path.basename(filename))[0]
            stl_path = os.path.join(input_dir, basename + ".STL")
            if not os.path.exists(stl_path):
                print(f"⚠️ STL not found: {stl_path}")
                continue

            try:
                mesh = trimesh.load(stl_path, force='mesh')
                prim_type, param = assign_primitive_by_link_name(link_name, mesh)

                # Replace mesh tag with primitive
                geometry_tag = collision.find("geometry")
                geometry_tag.remove(mesh_tag)

                if prim_type == "box":
                    new_shape = ET.SubElement(geometry_tag, "box")
                    new_shape.set("size", f"{param['size'][0]} {param['size'][1]} {param['size'][2]}")
                elif prim_type == "sphere":
                    new_shape = ET.SubElement(geometry_tag, "sphere")
                    new_shape.set("radius", f"{param['radius']}")
                elif prim_type == "cylinder":
                    new_shape = ET.SubElement(geometry_tag, "cylinder")
                    new_shape.set("radius", f"{param['radius']}")
                    new_shape.set("length", f"{param['length']}")

                # Copy origin from visual if missing
                if collision.find("origin") is None:
                    visual_origin = find_matching_visual_origin(link)
                    if visual_origin is not None:
                        origin_copy = ET.SubElement(collision, "origin")
                        for key, val in visual_origin.attrib.items():
                            origin_copy.set(key, val)

                print(f"✓ {link_name}: {filename} → <{prim_type}>")
            except Exception as e:
                print(f"⚠️ Failed {link_name} - {filename}: {e}")

    # Save
    output_path = urdf_path.replace(".urdf", "_primitive_named.urdf")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"\n✅ URDF saved to: {output_path}")


# Set paths
urdf_path = "./kistar_basic.urdf"
input_dir = "./meshes/kistar"

# Run
update_urdf_collision_with_named_primitives(urdf_path, input_dir)
