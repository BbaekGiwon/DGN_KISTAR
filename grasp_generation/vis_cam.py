"""Shared camera helper: derive a consistent palm-facing view from hand geometry.

Estimates the palm normal (thinnest PCA axis of the hand vertices) and the finger
axis (longest PCA axis), so every hand is shown palm-toward-camera, fingers-up,
regardless of its urdf/mjcf canonical frame. Also returns the centroid so callers
can center the geometry (fixes cropping under aspectmode='data')."""
import numpy as np


def palm_view(mesh_verts, contact_pts, dist=2.4):
    P = np.asarray(mesh_verts, dtype=float)
    C = P.mean(0)
    Pc = P - C
    if len(Pc) > 20000:
        idx = np.random.RandomState(0).choice(len(Pc), 20000, replace=False)
        Pc = Pc[idx]
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    finger = Vt[0]      # longest extent  -> finger direction
    normal = Vt[2]      # thinnest extent -> palm normal
    cc = np.asarray(contact_pts, dtype=float).mean(0) - C
    if np.dot(cc, normal) < 0:   # palm normal points toward the contact (palm) side
        normal = -normal
    if np.dot(cc, finger) < 0:   # finger axis points from palm toward fingertips
        finger = -finger
    eye = normal * dist
    cam = dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
               center=dict(x=0, y=0, z=0),
               up=dict(x=float(finger[0]), y=float(finger[1]), z=float(finger[2])))
    return C, cam
