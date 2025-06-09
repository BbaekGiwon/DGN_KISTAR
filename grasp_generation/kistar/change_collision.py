import os
import xml.etree.ElementTree as ET

def convert_stl_to_obj_path(stl_path: str) -> str:
    if stl_path.startswith("meshes/kistar/") and stl_path.endswith(".STL"):
        base_name = os.path.basename(stl_path).replace(".STL", ".obj")
        return f"coll_meshes/kistar/{base_name}"
    return stl_path  # 그대로 유지

def update_urdf_collision_paths(input_urdf_path, output_urdf_path):
    tree = ET.parse(input_urdf_path)
    root = tree.getroot()

    for collision in root.findall(".//collision"):
        mesh = collision.find(".//mesh")
        if mesh is not None and 'filename' in mesh.attrib:
            old_path = mesh.attrib['filename']
            new_path = convert_stl_to_obj_path(old_path)
            if old_path != new_path:
                print(f"Updating: {old_path}  →  {new_path}")
                mesh.set('filename', new_path)

    tree.write(output_urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"\n✅ Saved modified URDF to: {output_urdf_path}")

if __name__ == "__main__":
    input_urdf = "kistar_basic.urdf"  # 🔁 원본 URDF 경로
    output_urdf = "kistar_basic_modified.urdf"  # 📝 수정된 URDF 저장 경로
    update_urdf_collision_paths(input_urdf, output_urdf)
