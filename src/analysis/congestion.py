import numpy as np
from collections import deque


class CongestionDetector:
    name = 'congestion'

    def __init__(self, scale_mpp: float,
                 speed_threshold_kmh: float = 20.0,
                 density_threshold: int = 3,
                 window_s: float = 10.0):
        self.scale_mpp = scale_mpp
        self.speed_threshold_kmh = speed_threshold_kmh
        self.density_threshold = density_threshold
        self.window_s = window_s
        self._prev_pos = {}
        self._prev_frame = {}
        self._speed_buf = deque()   # (time_s, mean_speed_kmh)
        self._count_buf = deque()   # (time_s, count)
        self._is_congested = False
        self._cong_start_s = None
        self.events = []

    def update(self, frame_idx, track_data, fps):
        if fps <= 0:
            return
        t = frame_idx / fps
        speeds = []
        for td in track_data:
            tid = td['track_id']
            cx, cy = td['cx'], td['cy']
            if tid in self._prev_pos:
                px, py = self._prev_pos[tid]
                dt = (frame_idx - self._prev_frame[tid]) / fps
                if dt > 0:
                    dist_m = np.sqrt((cx - px) ** 2 + (cy - py) ** 2) * self.scale_mpp
                    speeds.append(dist_m / dt * 3.6)
            self._prev_pos[tid] = (cx, cy)
            self._prev_frame[tid] = frame_idx

        if speeds:
            self._speed_buf.append((t, float(np.mean(speeds))))
        self._count_buf.append((t, len(track_data)))

        cutoff = t - self.window_s
        while self._speed_buf and self._speed_buf[0][0] < cutoff:
            self._speed_buf.popleft()
        while self._count_buf and self._count_buf[0][0] < cutoff:
            self._count_buf.popleft()

        if self._speed_buf and self._count_buf:
            mean_speed = float(np.mean([s for _, s in self._speed_buf]))
            mean_count = float(np.mean([c for _, c in self._count_buf]))
            congested = (mean_speed < self.speed_threshold_kmh
                         and mean_count >= self.density_threshold)
            if congested and not self._is_congested:
                self._is_congested = True
                self._cong_start_s = t
            elif not congested and self._is_congested:
                self._is_congested = False
                self.events.append({
                    'start_s': round(self._cong_start_s, 2),
                    'end_s': round(t, 2),
                    'duration_s': round(t - self._cong_start_s, 2),
                    'mean_speed_kmh': round(mean_speed, 1),
                    'mean_object_count': round(mean_count, 1),
                })

    def finalize(self):
        return {'events': self.events}
