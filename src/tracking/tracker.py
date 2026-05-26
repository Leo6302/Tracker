from collections import deque

import numpy as np
from ultralytics import YOLO


class VideoTracker:
    def __init__(self, config, yolo_device: str = "cpu"):
        self.config = config
        self.yolo_device = yolo_device
        self.model = YOLO(config.get('yolo_model', 'yolo11n.pt'))
        self.seq_len = config['seq_len']
        self.conf_thresh = config['conf_thresh']
        # {track_id: deque of (frame_idx, cx, cy, w, h)}
        self.track_history: dict[int, deque] = {}
        self.track_classes: dict[int, str] = {}

    def reset(self):
        self.track_history.clear()
        self.track_classes.clear()

    def process_video(self, video_path, class_filter=None):
        """Yield (frame_idx, frame, track_data, (fw, fh), total_frames) per frame.

        track_data is a list of dicts:
          track_id, bbox_xyxy, cx, cy, w, h, cls_name, history, frame_w, frame_h
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.reset()

        results = self.model.track(
            str(video_path),
            tracker="bytetrack.yaml",
            conf=self.conf_thresh,
            stream=True,
            verbose=False,
            device=self.yolo_device,
        )

        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            fh, fw = frame.shape[:2]
            track_data = []

            if result.boxes is not None and result.boxes.id is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    cls_id = int(box.cls.item())
                    cls_name = self.model.names.get(cls_id, str(cls_id))

                    if class_filter and cls_name not in class_filter:
                        continue

                    x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
                    bw = x2 - x1
                    bh = y2 - y1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if track_id not in self.track_history:
                        self.track_history[track_id] = deque(maxlen=self.seq_len)
                    self.track_history[track_id].append(
                        (frame_idx, cx, cy, bw, bh)
                    )
                    self.track_classes[track_id] = cls_name

                    track_data.append({
                        'track_id': track_id,
                        'bbox_xyxy': (int(x1), int(y1), int(x2), int(y2)),
                        'cx': cx, 'cy': cy, 'w': bw, 'h': bh,
                        'cls_name': cls_name,
                        'history': self.track_history[track_id],
                        'frame_w': fw, 'frame_h': fh,
                    })

            yield frame_idx, frame, track_data, (fw, fh), total_frames
