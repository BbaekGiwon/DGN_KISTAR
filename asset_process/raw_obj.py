import os
import subprocess
import argparse
from tqdm import tqdm
import trimesh


def convert_stl_to_obj(stl_path, obj_path):
    mesh = trimesh.load(stl_path, force='mesh')
    simplified = mesh.simplify_quadratic_decimation(100)
    simplified.export(obj_path)


def run_coacd(coacd_bin, input_obj_path, output_dir,
            #   resolution=20_000, concavity=0.01,
              resolution=1, concavity=10,
              plane_downsampling=100, convex_hull_downsampling=100,
              alpha=10, beta=10, max_convex_hulls=1):
    assert os.path.exists(coacd_bin), f"CoACD binary not found at {coacd_bin}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        coacd_bin,
        "--input", input_obj_path,
        "--output", output_dir+".obj",
        "--resolution", str(resolution),
        "--concavity", str(concavity),
        "--planeDownsampling", str(plane_downsampling),
        "--convexhullDownsampling", str(convex_hull_downsampling),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--maxConvexHulls", str(max_convex_hulls)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Finished: {input_obj_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {input_obj_path}")
        print(e)


def batch_process_stl(coacd_bin, input_dir, output_dir, temp_dir="./tmp_obj"):
    os.makedirs(temp_dir, exist_ok=True)
    mesh_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.stl')]
    print(f"Found {len(mesh_files)} STL files in {input_dir}")

    for mesh_file in tqdm(mesh_files):
        stl_path = os.path.join(input_dir, mesh_file)
        name = os.path.splitext(mesh_file)[0]
        obj_path = os.path.join(temp_dir, name + ".obj")
        output_path = os.path.join(output_dir, name)

        convert_stl_to_obj(stl_path, obj_path)
        run_coacd(coacd_bin, obj_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--coacd_path", type=str, default="../thirdparty/CoACD/build/main",
                        help="Path to CoACD binary (e.g., ../CoACD/build/bin/CoACD)")
    parser.add_argument("--input_dir", type=str, default="../grasp_generation/kistar/meshes/kistar",
                        help="Input directory containing STL files")
    parser.add_argument("--output_dir", type=str, default="../grasp_generation/kistar/coll_meshes/kistar",
                        help="Output directory for convex-decomposed meshes")
    args = parser.parse_args()

    batch_process_stl(args.coacd_path, args.input_dir, args.output_dir)
