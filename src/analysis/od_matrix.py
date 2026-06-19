import numpy as np
import pandas as pd
import cv2


def _in_zone(cx, cy, polygon_pts):
    poly = np.array(polygon_pts, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


class ODMatrixBuilder:
    name = 'od_matrix'

    def __init__(self, zone_configs):
        self.zones = zone_configs          # [{'name': str, 'polygon': [[x,y],...]}]
        self._track_origin = {}            # {track_id: zone_name}
        self._track_in_zone = {}           # {track_id: zone_name or None}
        self.matrix = {}                   # {(origin, dest): int}

    def _get_zone(self, cx, cy):
        for z in self.zones:
            if _in_zone(cx, cy, z['polygon']):
                return z['name']
        return None

    def update(self, frame_idx, track_data, fps):
        for td in track_data:
            tid = td['track_id']
            cx, cy = td['cx'], td['cy']
            zone = self._get_zone(cx, cy)
            if zone is None:
                # Between zones — keep the track's last confirmed zone intact
                # so a gap (e.g. open road) between two zones doesn't erase it.
                continue
            prev = self._track_in_zone.get(tid)
            if tid not in self._track_origin:
                self._track_origin[tid] = zone
            if prev is not None and prev != zone:
                origin = self._track_origin.get(tid, prev)
                if origin != zone:
                    key = (origin, zone)
                    self.matrix[key] = self.matrix.get(key, 0) + 1
            self._track_in_zone[tid] = zone

    def finalize(self):
        zone_names = [z['name'] for z in self.zones]
        df = pd.DataFrame(0, index=zone_names, columns=zone_names)
        for (ori, dst), cnt in self.matrix.items():
            if ori in df.index and dst in df.columns:
                df.loc[ori, dst] = cnt
        return {
            'matrix_df': df,
            'raw': dict(self.matrix),
            'zone_names': zone_names,
        }
