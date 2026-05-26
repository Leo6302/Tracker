import numpy as np
import torch


def normalize_history(history, frame_w, frame_h):
    """Convert list of (frame_idx, cx, cy, w, h) tuples to normalised (N, 4) array."""
    arr = np.array([[p[1], p[2], p[3], p[4]] for p in history], dtype=np.float32)
    arr[:, 0] /= frame_w   # cx
    arr[:, 1] /= frame_h   # cy
    arr[:, 2] /= frame_w   # w
    arr[:, 3] /= frame_h   # h
    return arr


def build_sequences(history, seq_len, frame_w, frame_h):
    """Return (1, seq_len, 4) tensor ready for LSTM, or None if history is too short."""
    if len(history) < seq_len:
        return None
    coords = normalize_history(list(history)[-seq_len:], frame_w, frame_h)
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)


def denormalize(pred_norm, frame_w, frame_h):
    """Convert normalised (pred_len, 4) prediction back to pixel coordinates."""
    result = pred_norm.astype(np.float32).copy()
    result[:, 0] *= frame_w   # cx
    result[:, 1] *= frame_h   # cy
    result[:, 2] *= frame_w   # w
    result[:, 3] *= frame_h   # h
    return result.astype(np.int32)
