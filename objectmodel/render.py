"""Synthetic RGB-D capture of a 3D articulated object (z-buffered point splatting).

Surfaces carry a blocky procedural texture in part-local coordinates, so features
stick to the part as it moves. Produces a grayscale video plus a metric depth
stack for exercising the RGB-D path.

Status (experimental): on these renders the built-in Lucas-Kanade front end
recovers a hinge axis to within ~4° but drifts under perspective change and
under-estimates large swings; the validator correctly flags such results. Large
3D articulations need a long-range point tracker (e.g. CoTracker) whose tracks
are fed to `objectmodel fit`.
"""
from __future__ import annotations

import numpy as np

from .synthetic import SyntheticObject, animate, sample_box_surface


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])  # rows: camera x (right), y (down), z (forward)


TEXEL_M = 0.012  # texture cell size; blocky cells give corners (smooth stripes cause the aperture problem)


def _texture(local: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cells = np.floor(local / TEXEL_M).astype(np.int64)
    salt = rng.integers(1, 2**31, 3)
    h = (cells * np.array([73856093, 19349663, 83492791]) ^ salt).sum(axis=1)
    return (40 + (h % 176)).astype(float)


def render_rgbd(
    obj: SyntheticObject,
    video_path: str,
    depth_path: str,
    frames: int = 40,
    size: tuple[int, int] = (480, 360),
    fov_deg: float = 60.0,
    eye=(0.3, -1.25, 0.95),
    target=(0.0, 0.0, 0.45),
    density: int = 400_000,
    seed: int = 0,
) -> dict:
    import cv2

    rng = np.random.default_rng(seed)
    W, H = size
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    Rc = _look_at(np.asarray(eye, float), np.asarray(target, float))
    eye = np.asarray(eye, float)
    areas = [2 * (p.size[0] * p.size[1] + p.size[1] * p.size[2] + p.size[0] * p.size[2]) for p in obj.parts]
    clouds, shades, labels, normals = [], [], [], []
    for i, p in enumerate(obj.parts):
        n = int(density * areas[i] / sum(areas))
        pts = sample_box_surface(p.center, p.size, n, rng)
        local = (pts - np.asarray(p.center)) / (np.asarray(p.size) / 2)
        axis = np.argmax(np.abs(local), axis=1)  # which face each point lies on
        nrm = np.zeros_like(pts)
        nrm[np.arange(n), axis] = np.sign(local[np.arange(n), axis])
        clouds.append(pts)
        normals.append(nrm)
        shades.append(_texture(pts - np.asarray(p.center), rng))
        labels.append(np.full(n, i))
    base, shade, lab = np.concatenate(clouds), np.concatenate(shades), np.concatenate(labels)
    nbase = np.concatenate(normals)
    poses, states = animate(obj, frames, rng)
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (W, H), False)
    depth = np.zeros((frames, H, W), np.float32)
    for t in range(frames):
        pts = np.empty_like(base)
        nrm = np.empty_like(base)
        for i in range(len(obj.parts)):
            m = lab == i
            pts[m] = poses[t][i].apply(base[m])
            nrm[m] = nbase[m] @ poses[t][i].R.T
        cam = (pts - eye) @ Rc.T
        z = cam[:, 2]
        u = np.round(cam[:, 0] / z * f + W / 2).astype(int)
        v = np.round(cam[:, 1] / z * f + H / 2).astype(int)
        facing = np.einsum("ij,ij->i", nrm, eye - pts) > 0  # back-face culling
        ok = facing & (z > 0.05) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        order = np.argsort(-z[ok])  # far to near: nearer points overwrite
        uu, vv, zz, ss = u[ok][order], v[ok][order], z[ok][order], shade[ok][order]
        img = np.zeros((H, W), np.uint8)
        dep = np.zeros((H, W), np.float32)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):  # 2x2 splats close pinholes
            img[vv + dv, uu + du] = ss
            dep[vv + dv, uu + du] = zz
        # fill splat pinholes from neighbours (both channels) so the texture doesn't flicker
        k = np.ones((3, 3), np.uint8)
        for _ in range(2):
            hole = dep == 0
            img = np.where(hole, cv2.dilate(img, k), img)
            dep = np.where(hole, cv2.dilate(dep, k), dep)
        writer.write(img)
        depth[t] = dep
    writer.release()
    np.save(depth_path, depth)
    return {"states": states, "fov_deg": fov_deg, "camera_rotation": Rc}
