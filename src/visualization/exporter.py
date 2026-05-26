import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class VideoExporter:
    def __init__(self, output_path, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    def write_frame(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class CSVExporter:
    def __init__(self, pred_len: int):
        self.pred_len = pred_len
        self.rows: list[dict] = []

    def add(self, frame_idx, track_id, cls_name, cx, cy, w, h,
            pred=None, pred_std=None):
        row = {
            'frame': frame_idx, 'track_id': track_id, 'class': cls_name,
            'cx': round(cx, 2), 'cy': round(cy, 2),
            'w': round(w, 2), 'h': round(h, 2),
        }
        for i in range(self.pred_len):
            if pred is not None and i < len(pred):
                row[f'pred_cx_{i+1}'] = round(float(pred[i, 0]), 2)
                row[f'pred_cy_{i+1}'] = round(float(pred[i, 1]), 2)
            else:
                row[f'pred_cx_{i+1}'] = None
                row[f'pred_cy_{i+1}'] = None
            if pred_std is not None and i < len(pred_std):
                row[f'pred_std_cx_{i+1}'] = round(float(pred_std[i, 0]), 3)
                row[f'pred_std_cy_{i+1}'] = round(float(pred_std[i, 1]), 3)
        self.rows.append(row)

    def save(self, path):
        pd.DataFrame(self.rows).to_csv(path, index=False)


class SummaryImageExporter:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.histories: dict[int, list] = {}
        self.predictions: dict[int, list] = {}
        self.colors: dict[int, tuple] = {}

    def update(self, track_id: int, cx: float, cy: float,
               color_rgb: tuple, pred=None):
        if track_id not in self.histories:
            self.histories[track_id] = []
            self.colors[track_id] = color_rgb
        self.histories[track_id].append((cx, cy))
        if pred is not None:
            self.predictions[track_id] = pred

    def save(self, path, bg_frame=None):
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        if bg_frame is not None:
            bg_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            ax.imshow(bg_rgb, alpha=0.25, extent=[0, self.frame_w, self.frame_h, 0])

        ax.set_xlim(0, self.frame_w)
        ax.set_ylim(self.frame_h, 0)

        patches = []
        for tid, pts in self.histories.items():
            if len(pts) < 2:
                continue
            c = self.colors.get(tid, (1, 1, 1))
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '-', color=c, linewidth=1.5, alpha=0.85)
            ax.plot(xs[-1], ys[-1], 'o', color=c, markersize=5)
            if tid in self.predictions:
                pxs = [pts[-1][0]] + [p[0] for p in self.predictions[tid]]
                pys = [pts[-1][1]] + [p[1] for p in self.predictions[tid]]
                ax.plot(pxs, pys, '--', color=c, linewidth=1.5, alpha=0.6)
            patches.append(mpatches.Patch(color=c, label=f'#{tid}'))

        if patches:
            ax.legend(handles=patches, loc='upper right', fontsize=7,
                      facecolor='#2d2d2d', labelcolor='white', framealpha=0.8)

        ax.set_title('Trajectory Overview  (─ history  ╌ predicted)',
                     color='white', fontsize=13)
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
