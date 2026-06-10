"""
Flat viewer for ALL valid grasps (from precompute_valid.py) — one slider scrolls
through every valid grasp across all objects, each shown with its own object.

Usage (DGA env):
  /root/miniconda3/envs/DGA/bin/python tools/viser_valid.py --bundle ../data/vis_valid_v3.npy --port 8080
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

    entries = list(np.load(args.bundle, allow_pickle=True))
    n = len(entries)
    print(f"loaded {n} valid grasps")

    server = viser.ViserServer(host=args.host, port=args.port)
    info = server.gui.add_text('info', initial_value='')
    slider = server.gui.add_slider('valid_idx', min=0, max=max(n - 1, 0), step=1, initial_value=0)

    def show(_=None):
        e = entries[int(slider.value) % n]
        server.scene.add_mesh_simple('object', e['obj_v'], e['obj_f'],
                                     color=(239, 132, 167), opacity=0.85)
        server.scene.add_mesh_simple('hand', e['hand_v'], e['hand_f'],
                                     color=(102, 192, 255), opacity=0.85)
        info.value = f"{int(slider.value)%n + 1}/{n}   {e['object']}"

    slider.on_update(show)
    show()
    print(f"viser on http://{args.host}:{args.port}")
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
