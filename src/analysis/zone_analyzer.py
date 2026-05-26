import numpy as np
import cv2
from collections import defaultdict


def _in_zone(cx, cy, polygon_pts):
    poly = np.array(polygon_pts, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


class ZoneAnalyzer:
    name = 'zone_analysis'

    def __init__(self, zone_configs, zone_areas=None):
        self.zones = zone_configs               # [{'name': str, 'polygon': [[x,y],...]}]
        self.zone_areas = zone_areas or {}      # {zone_name: area_sqm}
        self._track_entry = defaultdict(dict)   # {track_id: {zone_name: entry_frame}}
        self._track_in_zone = {}                # {track_id: zone_name or None}
        self._track_cls = {}
        self.dwell_records = []
        self._occupancy = defaultdict(lambda: defaultdict(int))  # {zone: {frame: count}}
        self._fps = 1.0
        self._max_frame = 0

    def _get_zone(self, cx, cy):
        for z in self.zones:
            if _in_zone(cx, cy, z['polygon']):
                return z['name']
        return None

    def update(self, frame_idx, track_data, fps):
        self._fps = fps
        self._max_frame = max(self._max_frame, frame_idx)
        for td in track_data:
            tid = td['track_id']
            cx, cy = td['cx'], td['cy']
            self._track_cls[tid] = td['cls_name']
            zone = self._get_zone(cx, cy)
            prev = self._track_in_zone.get(tid)

            if zone != prev:
                if prev is not None and prev in self._track_entry[tid]:
                    entry_frame = self._track_entry[tid].pop(prev)
                    dwell_s = (frame_idx - entry_frame) / max(fps, 1e-6)
                    self.dwell_records.append({
                        'track_id': tid,
                        'class': td['cls_name'],
                        'zone': prev,
                        'entry_s': round(entry_frame / max(fps, 1e-6), 2),
                        'exit_s': round(frame_idx / max(fps, 1e-6), 2),
                        'dwell_s': round(dwell_s, 2),
                    })
                if zone is not None:
                    self._track_entry[tid][zone] = frame_idx

            if zone is not None:
                self._occupancy[zone][frame_idx] += 1
            self._track_in_zone[tid] = zone

    def occupancy_timeseries(self, zone_name):
        fps = max(self._fps, 1e-6)
        max_t = int(self._max_frame / fps) + 1
        counts = [0] * max_t
        for frame_idx, cnt in self._occupancy[zone_name].items():
            t = int(frame_idx / fps)
            if t < max_t:
                counts[t] = max(counts[t], cnt)
        return {'time_s': list(range(max_t)), 'count': counts}

    def space_utilization(self, zone_name):
        area = self.zone_areas.get(zone_name)
        if not area:
            return None
        counts = self.occupancy_timeseries(zone_name)['count']
        if not counts:
            return None
        return {
            'mean_density_per_sqm': round(float(np.mean(counts) / area), 4),
            'peak_density_per_sqm': round(float(np.max(counts) / area), 4),
            'zone_area_sqm': area,
        }

    def peak_windows(self, zone_name, window_s=60):
        occ = self.occupancy_timeseries(zone_name)
        counts = np.array(occ['count'], dtype=float)
        if len(counts) == 0:
            return []
        window = max(1, min(window_s, len(counts)))
        rolling = np.convolve(counts, np.ones(window) / window, mode='valid')
        peaks = []
        for _ in range(3):
            if len(rolling) == 0 or rolling.max() == 0:
                break
            best = int(np.argmax(rolling))
            peaks.append({
                'start_s': best,
                'end_s': best + window - 1,
                'mean_occupancy': round(float(rolling[best]), 1),
            })
            lo = max(0, best - window)
            hi = min(len(rolling), best + window)
            rolling[lo:hi] = 0
        return peaks

    def finalize(self):
        zone_summaries = []
        for z in self.zones:
            name = z['name']
            dwell_times = [r['dwell_s'] for r in self.dwell_records if r['zone'] == name]
            zone_summaries.append({
                'zone': name,
                'entry_count': len(dwell_times),
                'mean_dwell_s': round(float(np.mean(dwell_times)), 2) if dwell_times else 0,
                'max_dwell_s': round(float(np.max(dwell_times)), 2) if dwell_times else 0,
                'utilization': self.space_utilization(name),
                'peak_windows': self.peak_windows(name),
            })
        return {
            'dwell_records': self.dwell_records,
            'zone_summaries': zone_summaries,
            'occupancy_timeseries': {
                z['name']: self.occupancy_timeseries(z['name'])
                for z in self.zones
            },
        }
