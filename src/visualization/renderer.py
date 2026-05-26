import cv2
import numpy as np


def get_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for the given track_id."""
    h = (track_id * 67) % 180
    hsv = np.array([[[h, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


class FrameRenderer:
    def draw(self, frame: np.ndarray, track_data: list,
             predictions: dict) -> np.ndarray:
        """Annotate frame with bboxes, past trails, and predicted paths."""
        for td in track_data:
            tid = td['track_id']
            color = get_color(tid)
            x1, y1, x2, y2 = td['bbox_xyxy']

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label chip
            label = f"#{tid} {td['cls_name']}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # Past trajectory — solid line, fading with age
            history = list(td['history'])
            if len(history) > 1:
                pts = [(int(p[1]), int(p[2])) for p in history]
                n = len(pts)
                for i in range(1, n):
                    alpha = i / n
                    c = tuple(int(ch * alpha) for ch in color)
                    cv2.line(frame, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)

            # Predicted trajectory — dashed lighter line
            if tid in predictions:
                pred = predictions[tid]
                cur = (int(td['cx']), int(td['cy']))
                all_pts = [cur] + [(int(p[0]), int(p[1])) for p in pred]
                light = tuple(min(255, int(c * 1.4)) for c in color)
                for i in range(1, len(all_pts)):
                    p1 = np.array(all_pts[i - 1], dtype=float)
                    p2 = np.array(all_pts[i], dtype=float)
                    dist = int(np.linalg.norm(p2 - p1))
                    if dist == 0:
                        continue
                    for j in range(0, dist, 8):
                        if (j // 8) % 2 == 0:
                            t1, t2 = j / dist, min((j + 5) / dist, 1.0)
                            pt1 = tuple((p1 + t1 * (p2 - p1)).astype(int))
                            pt2 = tuple((p1 + t2 * (p2 - p1)).astype(int))
                            cv2.line(frame, pt1, pt2, light, 2, cv2.LINE_AA)

        return frame

    def draw_counting_lines(self, frame: np.ndarray, counting_line_set) -> None:
        for line in counting_line_set.lines:
            color = (0, 255, 255)  # cyan
            cv2.line(frame, (line.x1, line.y1), (line.x2, line.y2),
                     color, 2, cv2.LINE_AA)
            total = sum(line.counts.values())
            label = f"{line.label}: {total}"
            cx = (line.x1 + line.x2) // 2
            cy = (line.y1 + line.y2) // 2 - 12
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (cx - 2, cy - lh - 4),
                          (cx + lw + 2, cy + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def draw_zones(self, frame: np.ndarray, zone_analyzer) -> None:
        for z in zone_analyzer.zones:
            poly = np.array(z['polygon'], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], (0, 100, 200))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [poly], True, (0, 180, 255), 2, cv2.LINE_AA)
            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            zone_name = z['name']
            occ = sum(1 for tid, zn in zone_analyzer._track_in_zone.items()
                      if zn == zone_name)
            label = f"{zone_name} [{occ}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (cx - lw // 2 - 2, cy - lh - 4),
                          (cx + lw // 2 + 2, cy + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (cx - lw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
