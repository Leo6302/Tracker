import numpy as np


def _cross2d(ax, ay, bx, by):
    return ax * by - ay * bx


class CountingLine:
    def __init__(self, label, x1, y1, x2, y2):
        self.label = label
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
        self.counts = {}       # {cls_name: int}
        self.crossings = []    # [{frame, time_s, track_id, class, direction}]
        self._last_pos = {}    # {track_id: (cx, cy)}

    def _crossed(self, px1, py1, px2, py2):
        ax, ay = self.x2 - self.x1, self.y2 - self.y1
        d1 = _cross2d(ax, ay, px1 - self.x1, py1 - self.y1)
        d2 = _cross2d(ax, ay, px2 - self.x1, py2 - self.y1)
        s1 = 1 if d1 > 0 else (-1 if d1 < 0 else 0)
        s2 = 1 if d2 > 0 else (-1 if d2 < 0 else 0)
        return s1 != 0 and s2 != 0 and s1 != s2

    def _update(self, frame_idx, track_data, fps):
        for td in track_data:
            tid = td['track_id']
            cx, cy = td['cx'], td['cy']
            cls = td['cls_name']
            if tid in self._last_pos:
                px, py = self._last_pos[tid]
                if self._crossed(px, py, cx, cy):
                    self.counts[cls] = self.counts.get(cls, 0) + 1
                    ax, ay = self.x2 - self.x1, self.y2 - self.y1
                    d = _cross2d(ax, ay, cx - self.x1, cy - self.y1)
                    self.crossings.append({
                        'frame': frame_idx,
                        'time_s': round(frame_idx / max(fps, 1e-6), 2),
                        'track_id': tid,
                        'class': cls,
                        'direction': 'forward' if d > 0 else 'backward',
                    })
            self._last_pos[tid] = (cx, cy)

    def flow_rates(self, duration_s):
        if duration_s <= 0:
            return {}
        return {cls: round(cnt / duration_s * 3600, 1) for cls, cnt in self.counts.items()}

    def headway_analysis(self):
        results = {}
        by_cls = {}
        for c in self.crossings:
            by_cls.setdefault(c['class'], []).append(c['time_s'])
        for cls, times in by_cls.items():
            if len(times) < 2:
                continue
            hw = np.diff(sorted(times))
            results[cls] = {
                'mean_s': round(float(np.mean(hw)), 2),
                'std_s': round(float(np.std(hw)), 2),
                'min_s': round(float(np.min(hw)), 2),
                'max_s': round(float(np.max(hw)), 2),
            }
        return results


class CountingLineSet:
    name = 'counting_lines'

    def __init__(self, line_configs):
        self.lines = [CountingLine(**cfg) for cfg in line_configs]
        self._fps = 1.0
        self._total_frames = 0

    def update(self, frame_idx, track_data, fps):
        self._fps = fps
        self._total_frames = max(self._total_frames, frame_idx + 1)
        for line in self.lines:
            line._update(frame_idx, track_data, fps)

    def finalize(self):
        duration_s = self._total_frames / max(self._fps, 1e-6)
        return [
            {
                'label': line.label,
                'counts': line.counts,
                'crossings': line.crossings,
                'flow_rates_veh_hr': line.flow_rates(duration_s),
                'headway': line.headway_analysis(),
                'duration_s': round(duration_s, 2),
            }
            for line in self.lines
        ]

    def to_config_list(self):
        return [
            {'label': l.label, 'x1': l.x1, 'y1': l.y1, 'x2': l.x2, 'y2': l.y2}
            for l in self.lines
        ]
