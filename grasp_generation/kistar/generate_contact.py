import xml.etree.ElementTree as ET
import numpy as np
import json

def sample_box_points(size, n_points_z=3, n_points_y=2, fixed_x=None):
    """
    Box의 한 면(예: 손가락 앞면)에 균일 샘플링.
    - size: (x, y, z) 박스 크기
    - n_points_z: z축 방향 샘플 개수
    - n_points_y: y축 방향 샘플 개수
    - fixed_x: 박스 앞면(x) 좌표 (None이면 x/2 사용)
    """
    x, y, z = size
    # y, x, z = size
    # 앞면 기준, x/2 지점에서 샘플링
    px = x/2 if fixed_x is None else fixed_x
    # y, z 범위 샘플링
    y_samples = np.linspace(-y/4, y/4, n_points_y)
    z_samples = np.linspace(z*0.25, z*0.75, n_points_z)  # 양끝은 피함
    points = []
    for yy in y_samples:
        for zz in z_samples:
            points.append([px, yy, zz])
    return points


def sample_box_points_y(size, n_points_z=4, n_points_x=2, fixed_y=None):
    """
    Box의 한 면(예: 손가락 옆면)에 균일 샘플링.
    - size: (x, y, z) 박스 크기
    - n_points_z: z축 방향 샘플 개수
    - n_points_x: x축 방향 샘플 개수
    - fixed_y: 샘플링할 y 좌표 (None이면 y/2 사용)
    """
    x, y, z = size
    # 고정할 y 위치
    py = y/2 if fixed_y is None else fixed_y

    # x, z 범위 샘플링
    x_samples = np.linspace(-x/2, x/2, n_points_x)
    z_samples = np.linspace(z * 0.1, z * 0.9, n_points_z)  # 위·아래 끝점은 피함

    points = []
    for xx in x_samples:
        for zz in z_samples:
            # [x, y, z]
            points.append([xx, py, zz])
    return points


def sample_box_points_z(size, n_points_x=2, n_points_y=4, fixed_z=None):
    """
    Box의 한 면(예: 손가락 윗면)에 균일 샘플링.
    - size: (x, y, z) 박스 크기
    - n_points_x: x축 방향 샘플 개수
    - n_points_y: y축 방향 샘플 개수
    - fixed_z: 샘플링할 z 좌표 (None이면 z/2 사용)
    """
    x, y, z = size
    # 고정할 z 위치
    pz = z/2 if fixed_z is None else fixed_z

    # x, y 범위 샘플링
    x_samples = np.linspace(-x/2, x/2, n_points_x)
    y_samples = np.linspace(-y/2, y/2, n_points_y)

    points = []
    for xx in x_samples:
        for yy in y_samples:
            # [x, y, z]
            points.append([xx, yy, pz])
    return points  

def sample_tip_point(size):
    """_tip의 경우 단일 contact point (박스 앞면 중앙)"""
    x, y, z = size
    return [[-0.00190, 0,0.01626]]

def parse_urdf_and_generate_json(urdf_path, output_json):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    contact_dict = {}

    for link in root.findall("link"):
        name = link.attrib['name']
        # 기본값: 빈 리스트
        contact_points = []
        # collision geometry 추출
        collision = link.find("collision")
        if collision is not None:
            geometry = collision.find("geometry")
            if geometry is not None:
                box = geometry.find("box")
                if box is not None:
                    size_str = box.attrib['size']
                    size = [float(s) for s in size_str.split()]
                    contact_points = sample_box_points(size)

        if name in {"palm", "thumb_basemotor", "index_basemotor", "middle_basemotor", "ring_basemotor"}:
            contact_dict[name] = []
        elif name in {"index_tip", "middle_tip", "ring_tip", "thumb_tip"}:
            # contact_dict[name] = [[-0.00190, 0,0.01626]]
            contact_dict[name] = [[0.0052, 0.0, 0.01626]]
        else:
            contact_dict[name] = contact_points

    with open(output_json, "w") as f:
        json.dump(contact_dict, f, indent=2)
    print(f"Saved contact_points.json to {output_json}")

# 사용 예시
urdf_path = "./kistar.urdf"
output_json = "./contact_points.json"
parse_urdf_and_generate_json(urdf_path, output_json)
