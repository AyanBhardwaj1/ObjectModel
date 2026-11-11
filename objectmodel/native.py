"""Optional C++ core (cpp/twin_core.cpp) loaded with ctypes.

    cmake -S cpp -B cpp/build && cmake --build cpp/build
    python -m objectmodel.native          # parity check and timing against the NumPy versions

The Python package never requires it; these functions mirror geometry.kabsch,
joints._fit_revolute / _fit_prismatic and the DLS step in humanoid._ik_step.
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("OBJECTMODEL_NATIVE", ""),
    os.path.join(_HERE, "..", "cpp", "build", "libobjectmodel_core.so"),
    os.path.join(_HERE, "..", "cpp", "build", "libobjectmodel_core.dylib"),
]
_P = ctypes.POINTER(ctypes.c_double)
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    for path in _CANDIDATES:
        if path and os.path.exists(path):
            lib = ctypes.CDLL(os.path.abspath(path))
            lib.om_kabsch.argtypes = [_P, _P, _P, ctypes.c_int, _P, _P]
            lib.om_fit_revolute.argtypes = [_P, ctypes.c_int, _P, _P, _P, _P]
            lib.om_fit_prismatic.argtypes = [_P, ctypes.c_int, _P, _P]
            lib.om_dls_step.argtypes = [_P, ctypes.c_int, _P, ctypes.c_double, _P]
            for f in (lib.om_kabsch, lib.om_fit_revolute, lib.om_fit_prismatic, lib.om_dls_step):
                f.restype = ctypes.c_int
            lib.om_bench_dls.argtypes = [_P, ctypes.c_int, _P, ctypes.c_double, ctypes.c_int]
            lib.om_bench_dls.restype = ctypes.c_double
            _lib = lib
            return lib
    return None


def available() -> bool:
    return _load() is not None


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_P)


def _need():
    lib = _load()
    if lib is None:
        raise RuntimeError("native core not built: cmake -S cpp -B cpp/build && cmake --build cpp/build")
    return lib


def _check(code: int) -> None:
    if code != 0:
        raise ValueError(f"native core rejected the arguments (code {code})")


def _pack(motions) -> np.ndarray:
    """List of Pose -> contiguous (T, 12) array [R row-major, t]."""
    return np.ascontiguousarray(np.array([np.concatenate([m.R.ravel(), m.t]) for m in motions], float))


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    lib = _need()
    P = np.ascontiguousarray(P, float)
    Q = np.ascontiguousarray(Q, float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("kabsch expects matching (N,3) arrays")
    if w is not None and np.shape(w) != (len(P),):
        raise ValueError("w must have one weight per point")
    wp = None if w is None else _ptr(np.ascontiguousarray(w, float))
    R, t = np.zeros(9), np.zeros(3)
    _check(lib.om_kabsch(_ptr(P), _ptr(Q), wp, len(P), _ptr(R), _ptr(t)))
    return R.reshape(3, 3), t


def fit_revolute(motions, centroid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lib = _need()
    m = _pack(motions)
    axis, pivot, theta = np.zeros(3), np.zeros(3), np.zeros(len(m))
    c = np.ascontiguousarray(centroid, float)
    if c.shape != (3,):
        raise ValueError("centroid must be a 3-vector")
    _check(lib.om_fit_revolute(_ptr(m), len(m), _ptr(c), _ptr(axis), _ptr(pivot), _ptr(theta)))
    return axis, pivot, theta


def fit_prismatic(motions) -> tuple[np.ndarray, np.ndarray]:
    lib = _need()
    m = _pack(motions)
    axis, d = np.zeros(3), np.zeros(len(m))
    _check(lib.om_fit_prismatic(_ptr(m), len(m), _ptr(axis), _ptr(d)))
    return axis, d


def dls_step(J: np.ndarray, err: np.ndarray, lam: float) -> np.ndarray:
    lib = _need()
    J = np.ascontiguousarray(J, float)
    if J.ndim != 2 or J.shape[0] != 3:
        raise ValueError("J must be 3 x n")
    if np.shape(err) != (3,):
        raise ValueError("err must be a 3-vector")
    dq = np.zeros(J.shape[1])
    _check(lib.om_dls_step(_ptr(J), J.shape[1], _ptr(np.ascontiguousarray(err, float)), float(lam), _ptr(dq)))
    return dq


def _bench() -> None:
    from . import synthetic as s
    from .joints import _fit_revolute, relative_motions
    from .segmentation import segmentation_from_labels

    obj = s.laptop()
    ts = s.generate_tracks(obj, n_frames=240, seed=0)
    seg = segmentation_from_labels(ts, ts.labels)
    motions = relative_motions(seg.poses[0], seg.poses[1])
    c = ts.points[0, ts.labels == 1].mean(axis=0)

    def timeit(fn, n):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e6

    u_py, p_py, th_py = _fit_revolute(motions, c)
    u_cc, p_cc, th_cc = fit_revolute(motions, c)
    print(f"revolute fit, 240 frames: axis diff {np.abs(u_py - u_cc).max():.2e}, pivot diff {np.abs(p_py - p_cc).max():.2e}, "
          f"angle diff {np.abs(th_py - th_cc).max():.2e}")
    packed = _pack(motions)
    lib = _need()
    a, p, th = np.zeros(3), np.zeros(3), np.zeros(len(packed))
    cc = np.ascontiguousarray(c)
    t_py = timeit(lambda: _fit_revolute(motions, c), 50)
    t_cc = timeit(lambda: lib.om_fit_revolute(_ptr(packed), len(packed), _ptr(cc), _ptr(a), _ptr(p), _ptr(th)), 500)
    print(f"  NumPy {t_py:9.1f} µs   C++ {t_cc:9.1f} µs   x{t_py / t_cc:.0f}")
    J = np.random.default_rng(0).normal(size=(3, 9))
    e = np.array([0.01, -0.02, 0.005])
    dq = np.zeros(9)
    t_np = timeit(lambda: J.T @ np.linalg.solve(J @ J.T + 0.03 ** 2 * np.eye(3), e), 2000)
    t_c = timeit(lambda: lib.om_dls_step(_ptr(J), 9, _ptr(e), 0.03, _ptr(dq)), 20000)
    inner = lib.om_bench_dls(_ptr(J), 9, _ptr(e), 0.03, 200000) / 1000.0
    print(f"DLS IK step (3x9): NumPy {t_np:.2f} µs   C++ via ctypes {t_c:.2f} µs (call overhead)   "
          f"C++ in-loop {inner:.3f} µs   (a 1 kHz control loop has 1000 µs)")


if __name__ == "__main__":
    if not available():
        raise SystemExit("native core not built: cmake -S cpp -B cpp/build && cmake --build cpp/build")
    _bench()
