"""
DRO-Grasp-style viser viewer for KISTAR grasps (run in an env that has viser,
e.g. `dro` or `DGA`). Loads an .npz bundle produced by precompute_vis.py
(hand/object meshes already baked, so no torch/pk needed here).

Usage:
  /root/miniconda3/envs/DGA/bin/python tools/viser_vis.py --bundle ../data/vis_mouse.npz --port 8080
Then open http://<host>:8080
"""
import argparse, time
import numpy as np
import viser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()

    d = np.load(args.bundle, allow_pickle=True)
    obj_v, obj_f = d['obj_v'], d['obj_f']
    hand_f, hand_v = d['hand_f'], d['hand_v']     # hand_v: (N, V, 3)
    scales = d['scales']
    labels = d['labels']
    n = hand_v.shape[0]
    print(f"loaded {n} grasps for {str(d['object_code'])}")

    server = viser.ViserServer(host=args.host, port=args.port)

    def show(idx):
        idx = int(idx) % n
        s = float(scales[idx])
        server.scene.add_mesh_simple(
            'object', obj_v * s, obj_f,
            color=(239, 132, 167), opacity=0.85)
        server.scene.add_mesh_simple(
            'hand', hand_v[idx], hand_f,
            color=(102, 192, 255), opacity=0.85)
        info.value = f"{idx+1}/{n}   {labels[idx]}"

    info = server.gui.add_text('info', initial_value='')
    slider = server.gui.add_slider('grasp_idx', min=0, max=max(n - 1, 0),
                                   step=1, initial_value=0)
    slider.on_update(lambda _: show(slider.value))
    show(0)

    print(f"viser on http://{args.host}:{args.port}")
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
