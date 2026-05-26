import numpy as np
import cv2


class PerspectiveCalibrator:
    def __init__(self, mode='none'):
        self.mode = mode
        self.scale_mpp = None       # meters per pixel (reference mode)
        self.H = None               # homography matrix (4-point mode)
        self.residual_error = None

    def calibrate_reference(self, real_m: float, pixel_dist: float):
        if pixel_dist <= 0:
            return
        self.scale_mpp = real_m / pixel_dist
        self.mode = 'reference'

    def calibrate_homography(self, src_pts, real_w_m: float, real_h_m: float):
        """Compute homography from 4 pixel points to a known real_w × real_h rectangle."""
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array([
            [0, 0], [real_w_m, 0], [real_w_m, real_h_m], [0, real_h_m]
        ], dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst)
        self.mode = 'homography'
        projected = cv2.perspectiveTransform(src.reshape(1, -1, 2), self.H)
        self.residual_error = float(np.mean(np.linalg.norm(projected[0] - dst, axis=1)))
        # Approximate scale (average of width and height scale factors)
        self.scale_mpp = (real_w_m / max(np.linalg.norm(src[1] - src[0]), 1e-6) +
                          real_h_m / max(np.linalg.norm(src[3] - src[0]), 1e-6)) / 2

    def pixel_to_meters(self, px, py):
        if self.mode == 'reference' and self.scale_mpp:
            return px * self.scale_mpp, py * self.scale_mpp
        if self.mode == 'homography' and self.H is not None:
            pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
            res = cv2.perspectiveTransform(pt, self.H)
            return float(res[0, 0, 0]), float(res[0, 0, 1])
        return px, py

    def to_dict(self):
        return {
            'mode': self.mode,
            'scale_mpp': self.scale_mpp,
            'H': self.H.tolist() if self.H is not None else None,
            'residual_error': self.residual_error,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(d.get('mode', 'none'))
        obj.scale_mpp = d.get('scale_mpp')
        if d.get('H'):
            obj.H = np.array(d['H'], dtype=np.float64)
        obj.residual_error = d.get('residual_error')
        return obj
