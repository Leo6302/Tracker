import numpy as np
from collections import defaultdict


class SpeedEstimator:
    name = 'speed'

    def __init__(self, scale_mpp: float, ema_alpha: float = 0.3):
        self.scale_mpp = scale_mpp
        self.ema_alpha = ema_alpha
        self._prev_pos = {}             # {track_id: (cx, cy)}
        self._prev_frame = {}           # {track_id: frame_idx}
        self._smoothed = {}             # {track_id: speed_kmh}
        self.track_speeds = defaultdict(list)   # {track_id: [speed_kmh]}
        self.track_cls = {}             # {track_id: cls_name}

    def update(self, frame_idx, track_data, fps):
        if fps <= 0:
            return
        for td in track_data:
            tid = td['track_id']
            cx, cy = td['cx'], td['cy']
            self.track_cls[tid] = td['cls_name']
            if tid in self._prev_pos:
                px, py = self._prev_pos[tid]
                dt = (frame_idx - self._prev_frame[tid]) / fps
                if dt > 0:
                    dist_m = np.sqrt((cx - px) ** 2 + (cy - py) ** 2) * self.scale_mpp
                    raw_spd = dist_m / dt * 3.6  # km/h
                    prev = self._smoothed.get(tid, raw_spd)
                    smoothed = self.ema_alpha * raw_spd + (1 - self.ema_alpha) * prev
                    self._smoothed[tid] = smoothed
                    self.track_speeds[tid].append(smoothed)
            self._prev_pos[tid] = (cx, cy)
            self._prev_frame[tid] = frame_idx

    def finalize(self):
        by_class = defaultdict(list)
        for tid, speeds in self.track_speeds.items():
            cls = self.track_cls.get(tid, 'unknown')
            by_class[cls].extend(speeds)

        per_class = {}
        for cls, speeds in by_class.items():
            if not speeds:
                continue
            arr = np.array(speeds)
            per_class[cls] = {
                'count': len(arr),
                'mean': round(float(np.mean(arr)), 1),
                'std': round(float(np.std(arr)), 1),
                'p15': round(float(np.percentile(arr, 15)), 1),
                'p50': round(float(np.percentile(arr, 50)), 1),
                'p85': round(float(np.percentile(arr, 85)), 1),
                'p95': round(float(np.percentile(arr, 95)), 1),
                'max': round(float(np.max(arr)), 1),
            }

        return {
            'per_class': per_class,
            'track_speeds': dict(self.track_speeds),
            'track_cls': dict(self.track_cls),
            'scale_mpp': self.scale_mpp,
        }

    def get_current_speed(self, track_id):
        return self._smoothed.get(track_id)
