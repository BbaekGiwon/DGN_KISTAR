"""
Browse all objects' grasps in one viser viewer: object dropdown + grasp slider.
Loads vis_<object>.npz bundles (from precompute_all.py) lazily.

Usage (DGA env):
  /root/miniconda3/envs/DGA/bin/python tools/viser_vis_all.py --dir ../data/vis_all --port 8080
"""
import argparse, glob, os, time
import numpy as np
import viser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, '*.npz')))
    objects = [os.path.basename(f)[:-4] for f in files]
    print(f"{len(objects)} objects in {args.dir}")
    cache = {}

    def load(obj):
        if obj not in cache:
            cache[obj] = np.load(os.path.join(args.dir, obj + '.npz'), allow_pickle=True)
        return cache[obj]

    server = viser.ViserServer(host=args.host, port=args.port)
    dropdown = server.gui.add_dropdown('object', options=objects, initial_value=objects[0])
    slider = server.gui.add_slider('grasp_idx', min=0, max=1, step=1, initial_value=0)
    info = server.gui.add_text('info', initial_value='')

    def show():
        obj = dropdown.value
        d = load(obj)
        n = d['hand_v'].shape[0]
        idx = int(slider.value) % n
        server.scene.add_mesh_simple('object', d['obj_v'], d['obj_f'],
                                     color=(239, 132, 167), opacity=0.85)
        server.scene.add_mesh_simple('hand', d['hand_v'][idx], d['hand_f'],
                                     color=(102, 192, 255), opacity=0.85)
        info.value = f"{obj}  {idx+1}/{n}   {d['labels'][idx]}"

    def on_object(_):
        d = load(dropdown.value)
        slider.max = max(d['hand_v'].shape[0] - 1, 0)
        slider.value = 0
        show()

    dropdown.on_update(on_object)
    slider.on_update(lambda _: show())
    on_object(None)

    print(f"viser on http://{args.host}:{args.port}")
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
