import json
import hashlib
import datetime
from dataclasses import dataclass, field, asdict


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        except ImportError:
            pass
        return super().default(obj)


@dataclass
class SessionConfig:
    video_hash: str = ''
    yolo_model: str = 'yolo11n.pt'
    conf_thresh: float = 0.5
    seq_len: int = 20
    pred_len: int = 10
    lstm_mode: str = 'pretrained'
    class_filter: list = field(default_factory=list)
    counting_lines: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    zone_areas: dict = field(default_factory=dict)
    scale_mpp: float = None
    enable_traffic: bool = False
    enable_speed: bool = False
    enable_od: bool = False
    enable_urban: bool = False
    enable_heatmap: bool = False
    heatmap_classes: list = field(default_factory=lambda: ['person'])
    export_format: str = 'CSV만'
    enable_mc_dropout: bool = False
    chart_format: str = 'HTML (인터랙티브)'
    calibration: dict = field(default_factory=dict)
    created_at: str = ''
    notes: str = ''
    stats: dict = field(default_factory=dict)


class AnalysisSession:
    @staticmethod
    def compute_video_hash(video_path) -> str:
        try:
            h = hashlib.sha256()
            with open(str(video_path), 'rb') as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except Exception:
            return ''

    @staticmethod
    def save(path, config: SessionConfig):
        config.created_at = datetime.datetime.now().isoformat()
        with open(str(path), 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

    @staticmethod
    def load(path) -> SessionConfig:
        with open(str(path), 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = SessionConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
