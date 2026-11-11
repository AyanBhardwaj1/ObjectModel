"""Video front end: frames -> 2D point tracks -> metric 3D tracks.

* Tracking: Shi-Tomasi corners on frame 0, pyramidal Lucas-Kanade frame to frame,
  forward-backward consistency check; lost tracks become invisible (identity is
  always relative to frame 0, which defines the rest pose).
* Background: static tracks far from anything that moves are dropped (auto ROI),
  so a textured table doesn't get welded to the object's base.
* Lifting: with a depth stack (T,H,W metres) points are back-projected through a
  pinhole camera; without depth we assume motion in the image plane (side-on
  capture) and set metric scale from one known dimension (--scale-mm).
Upstream trackers (CoTracker, SAM-mask tracks + metric depth) can bypass this
module entirely by writing a tracks .npz for `objectmodel fit`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synthetic import TrackSet

FB_MAX_PX = 1.0
MOVING_PX = 3.0
ROI_GROW = 0.35
DEFAULT_SCENE_WIDTH_M = 0.5


@dataclass(frozen=True)
class VideoConfig:
    max_frames: int = 120
    stride: int = 1
    max_corners: int = 1500
    scale_mm: float | None = None  # known width (mm) of the moving object's extent in frame 0
    marker_mm: float | None = None  # side length (mm) of a printed ArUco 4x4 marker in view
    depth: str | None = None  # path to .npy (T,H,W) metres
    fov_deg: float = 60.0  # horizontal field of view for depth back-projection
    keep_background: bool = False


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ValueError("video input needs OpenCV: pip install opencv-python-headless") from exc
    return cv2


def read_frames(path: str, cfg: VideoConfig) -> list[np.ndarray]:
    cv2 = _cv2()
    if cfg.stride < 1 or cfg.max_frames < 3:
        raise ValueError("stride must be >= 1 and max_frames >= 3")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    frames, i = [], 0
    while len(frames) < cfg.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % cfg.stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame)
        i += 1
    cap.release()
    if len(frames) < 3:
        raise ValueError("video has fewer than 3 usable frames")
    return frames


def track_points(frames: list[np.ndarray], cfg: VideoConfig) -> tuple[np.ndarray, np.ndarray]:
    """-> uv (T,N,2) pixels, visible (T,N)."""
    cv2 = _cv2()
    p0 = cv2.goodFeaturesToTrack(frames[0], cfg.max_corners, 0.01, 4, blockSize=5)
    if p0 is None or len(p0) < 20:
        raise ValueError("too few trackable features; add texture or light the object better")
    T, N = len(frames), len(p0)
    uv = np.zeros((T, N, 2), np.float32)
    vis = np.zeros((T, N), bool)
    uv[0], vis[0] = p0[:, 0], True
    lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for t in range(1, T):
        prev = uv[t - 1].reshape(-1, 1, 2)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(frames[t - 1], frames[t], prev, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(frames[t], frames[t - 1], nxt, None, **lk)
        fb = np.linalg.norm(back[:, 0] - prev[:, 0], axis=1)
        ok = vis[t - 1] & (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < FB_MAX_PX)
        uv[t] = np.where(ok[:, None], nxt[:, 0], uv[t - 1])
        vis[t] = ok
    return uv, vis


def drop_background(uv: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Boolean mask of tracks to keep: moving tracks + static tracks near them."""
    disp = np.where(vis[..., None], uv - uv[0], 0)
    moving = np.linalg.norm(disp, axis=2).max(axis=0) > MOVING_PX
    if moving.sum() < 10:
        return np.ones(len(moving), bool)
    pts = uv[:, moving][vis[:, moving]]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    grow = ROI_GROW * (hi - lo)
    lo, hi = lo - grow, hi + grow
    inside = np.all((uv[0] >= lo) & (uv[0] <= hi), axis=1)
    return moving | inside


def marker_scale(frame: np.ndarray, marker_mm: float) -> float:
    """Metres per pixel from an ArUco (DICT_4X4_50) marker of known side length."""
    cv2 = _cv2()
    if marker_mm <= 0:
        raise ValueError("marker size must be positive")
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None or len(corners) == 0:
        raise ValueError("no ArUco 4x4 marker found in the first frame")
    c = corners[0][0]
    side_px = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    return marker_mm / 1000.0 / side_px


def lift_planar(uv: np.ndarray, vis: np.ndarray, scale_mm: float | None, width_px: int,
                m_per_px: float | None = None) -> np.ndarray:
    """Image-plane motion -> metres in a z-up frame (x right, z up, y into the scene)."""
    if m_per_px:
        pass
    elif scale_mm:
        extent = np.ptp(uv[0][vis[0]][:, 0])
        m_per_px = scale_mm / 1000.0 / max(extent, 1.0)
    else:
        m_per_px = DEFAULT_SCENE_WIDTH_M / width_px
    out = np.zeros(uv.shape[:2] + (3,))
    out[..., 0] = uv[..., 0] * m_per_px
    out[..., 2] = -uv[..., 1] * m_per_px
    return out


def lift_depth(uv: np.ndarray, vis: np.ndarray, depth: np.ndarray, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    T, H, W = depth.shape
    if T < uv.shape[0]:
        raise ValueError("depth stack has fewer frames than the video")
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    cx, cy = W / 2, H / 2
    out = np.zeros(uv.shape[:2] + (3,))
    vis = vis.copy()
    for t in range(uv.shape[0]):
        u = np.clip(np.round(uv[t, :, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(uv[t, :, 1]).astype(int), 0, H - 1)
        z = depth[t, v, u]
        good = np.isfinite(z) & (z > 0)
        vis[t] &= good
        z = np.where(good, z, 0.0)
        # camera (x right, y down, z forward) -> z-up world (x right, y forward, z up)
        out[t, :, 0] = (uv[t, :, 0] - cx) * z / f
        out[t, :, 1] = z
        out[t, :, 2] = -(uv[t, :, 1] - cy) * z / f
    return out, vis


def tracks_from_video(path: str, cfg: VideoConfig = VideoConfig()) -> TrackSet:
    frames = read_frames(path, cfg)
    uv, vis = track_points(frames, cfg)
    keep = np.ones(uv.shape[1], bool) if cfg.keep_background else drop_background(uv, vis)
    uv, vis = uv[:, keep], vis[:, keep]
    if cfg.depth:
        depth = np.load(cfg.depth, allow_pickle=False)
        pts, vis = lift_depth(uv, vis, depth[:: cfg.stride], cfg.fov_deg)
    else:
        mpp = marker_scale(frames[0], cfg.marker_mm) if cfg.marker_mm else None
        pts = lift_planar(uv, vis, cfg.scale_mm, frames[0].shape[1], mpp)
    # drop tracks that die almost immediately; they carry no motion information
    alive = vis.mean(axis=0) > 0.5
    pts, vis = pts[:, alive], vis[:, alive]
    vis[0] = True
    return TrackSet(np.where(vis[..., None], pts, 0.0), vis, meta={"source": path, "planar": not cfg.depth})


# ------------------------------------------------------------ synthetic video

def render_synthetic_video(path: str, kind: str = "laptop", frames: int = 60, size=(480, 360), seed: int = 0,
                           marker_px: int = 0) -> dict:
    """Side-on render of a 2D articulated object with textured parts. Returns ground truth."""
    cv2 = _cv2()
    rng = np.random.default_rng(seed)
    W, H = size

    def texture(w, h):
        img = np.full((h, w), 140, np.uint8)
        for _ in range(w * h // 120):
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.circle(img, c, int(rng.integers(2, 5)), int(rng.integers(0, 255)), -1)
        return cv2.GaussianBlur(img, (3, 3), 0)

    bg = cv2.GaussianBlur(texture(W, H), (0, 0), 1.0) // 2 + 40
    if marker_px:
        mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 3, marker_px)
        pad = marker_px // 5
        bg[10:10 + marker_px + 2 * pad, 10:10 + marker_px + 2 * pad] = 255
        bg[10 + pad:10 + pad + marker_px, 10 + pad:10 + pad + marker_px] = mk
    if kind == "laptop":
        base_rect = (90, 250, 300, 22)  # x, y, w, h (px)
        hinge = np.array([388.0, 250.0])
        lid_size = (40, 200)  # w, h; lid stands up from the hinge
        child_tex = texture(*lid_size)
        q = np.radians(70) * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, frames)))
    elif kind == "drawer":
        base_rect = (110, 80, 260, 240)
        child_tex = texture(150, 60)
        q = 110 * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, frames)))
    else:
        raise ValueError(f"unknown synthetic video kind {kind!r}")
    base_tex = texture(base_rect[2], base_rect[3])
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, 20.0, (W, H), False)
    if not writer.isOpened():
        raise ValueError("could not open video writer")
    for t in range(frames):
        img = bg.copy()
        x, y, w, h = base_rect
        img[y:y + h, x:x + w] = base_tex
        if kind == "laptop":
            # lid occupies [-w, 0] x [-h, 0] relative to the hinge, rotated by -q (opens toward +x)
            M = cv2.getRotationMatrix2D((float(lid_size[0]), float(lid_size[1])), -np.degrees(q[t]), 1.0)
            M[:, 2] += hinge - np.array([lid_size[0], lid_size[1]], float)
            warped = cv2.warpAffine(child_tex, M, (W, H), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(np.full(child_tex.shape, 255, np.uint8), M, (W, H))
        else:
            M = np.float32([[1, 0, 160 + q[t]], [0, 1, 150]])
            warped = cv2.warpAffine(child_tex, M, (W, H))
            mask = cv2.warpAffine(np.full(child_tex.shape, 255, np.uint8), M, (W, H))
        img = np.where(mask > 127, warped, img)
        writer.write(img)
    writer.release()
    if kind == "laptop":
        return {"type": "revolute", "axis_image": "normal", "pivot_px": hinge.tolist(), "range_rad": float(q.max()),
                "base_width_px": base_rect[2]}
    return {"type": "prismatic", "direction_px": [1, 0], "range_px": float(q.max()), "base_width_px": base_rect[2]}
