import numpy as np
import cv2


class DensityHeatmapBuilder:
    name = 'heatmap'

    def __init__(self, frame_w: int, frame_h: int, target_classes=None):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.target_classes = set(target_classes) if target_classes else None
        self.accumulator = np.zeros((frame_h, frame_w), dtype=np.float32)

    def update(self, frame_idx, track_data, fps):
        for td in track_data:
            if self.target_classes and td['cls_name'] not in self.target_classes:
                continue
            cx, cy = int(round(td['cx'])), int(round(td['cy']))
            w, h = td['w'], td['h']
            if not (0 <= cx < self.frame_w and 0 <= cy < self.frame_h):
                continue
            sigma = max(int(max(w, h) / 3), 5)
            x0 = max(0, cx - 3 * sigma)
            x1 = min(self.frame_w, cx + 3 * sigma + 1)
            y0 = max(0, cy - 3 * sigma)
            y1 = min(self.frame_h, cy + 3 * sigma + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            xg = np.arange(x0, x1) - cx
            yg = np.arange(y0, y1) - cy
            xx, yy = np.meshgrid(xg, yg)
            kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2)).astype(np.float32)
            self.accumulator[y0:y1, x0:x1] += kernel

    def finalize(self):
        return {'accumulator': self.accumulator.copy()}

    def save_heatmap(self, path, bg_frame=None):
        acc = self.accumulator.copy()
        norm = (acc / acc.max() * 255).astype(np.uint8) if acc.max() > 0 else acc.astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        if bg_frame is not None:
            alpha = (norm.astype(np.float32) / 255.0) ** 0.5
            alpha = alpha[:, :, None]
            result = (bg_frame.astype(np.float32) * (1 - alpha)
                      + colored.astype(np.float32) * alpha).astype(np.uint8)
        else:
            result = colored
        cv2.imwrite(str(path), result)
