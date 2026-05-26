import sys
import json
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import torch
import yaml
import gradio as gr

from src.pipeline import TrackingPipeline
from src.prediction.trainer import pretrain_model
from src.device_utils import detect_devices
from src.analysis.session import AnalysisSession, SessionConfig

import psutil
from collections import deque
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import GPUtil as _gputil_mod
    _GPUTIL_AVAILABLE = True
except Exception:
    _gputil_mod = None
    _GPUTIL_AVAILABLE = False

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PRETRAINED_PATH = Path(__file__).parent / "models" / "lstm_pretrained.pt"

with open(CONFIG_PATH) as f:
    BASE_CONFIG: dict = yaml.safe_load(f)

YOLO_DEVICE, LSTM_DEVICE, DEVICE_DESC = detect_devices()
print(f"[장치] {DEVICE_DESC}")

if not PRETRAINED_PATH.exists():
    print("사전학습 모델 없음 — 생성 중 (첫 실행 시 1~2분 소요)")
    pretrain_model(BASE_CONFIG, PRETRAINED_PATH)

COCO_TRACKABLE = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow",
]

DEFAULT_LINES = '[{"label": "Line 1", "x1": 100, "y1": 300, "x2": 800, "y2": 300}]'
DEFAULT_ZONES = '[]'
DEFAULT_ZONE_AREAS = '{}'

_STATS_HISTORY: deque = deque(maxlen=60)
psutil.cpu_percent(interval=None)  # prime first measurement


def _safe_json(text, default):
    try:
        return json.loads(text)
    except Exception:
        return default


def _collect_stats() -> dict:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    gpu_load, gpu_mem, has_gpu = 0.0, 0.0, False
    if _GPUTIL_AVAILABLE:
        try:
            gpus = _gputil_mod.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100
                gpu_mem = gpus[0].memoryUtil * 100
                has_gpu = True
        except Exception:
            pass
    entry = {
        'cpu': cpu,
        'ram': mem.percent,
        'ram_gb': mem.used / 1e9,
        'gpu': gpu_load,
        'gpu_mem': gpu_mem,
        'has_gpu': has_gpu,
    }
    _STATS_HISTORY.append(entry)
    return entry


def _make_chart(values: list, ylabel: str, color: str):
    fig, ax = plt.subplots(figsize=(4, 2))
    xs = list(range(len(values)))
    ax.plot(xs, values, color=color, linewidth=1.5)
    ax.fill_between(xs, values, alpha=0.15, color=color)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.25, axis='y')
    for spine in ('top', 'right', 'bottom'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.4)
    return fig


def _update_monitor():
    entry = _collect_stats()
    gpu_str = f"GPU {entry['gpu']:.0f}%" if entry['has_gpu'] else "GPU: N/A"
    summary = (
        f"CPU {entry['cpu']:.0f}%  |  "
        f"RAM {entry['ram_gb']:.1f} GB ({entry['ram']:.0f}%)  |  "
        f"{gpu_str}"
    )
    if len(_STATS_HISTORY) < 2:
        return summary, None, None, None
    hist = list(_STATS_HISTORY)
    cpu_fig = _make_chart([e['cpu'] for e in hist], 'CPU %', '#4472C4')
    ram_fig = _make_chart([e['ram'] for e in hist], 'RAM %', '#70AD47')
    gpu_fig = (
        _make_chart([e['gpu'] for e in hist], 'GPU %', '#ED7D31')
        if entry['has_gpu'] else None
    )
    return summary, cpu_fig, ram_fig, gpu_fig


def _resolve_file_paths(files) -> list:
    """Gradio 6.x gr.File multiple이 반환하는 FileData 객체에서 경로를 추출한다."""
    paths = []
    for f in (files or []):
        if isinstance(f, str):
            paths.append(f)
        elif isinstance(f, dict):
            paths.append(f.get('path') or f.get('name', ''))
        elif hasattr(f, 'path'):
            paths.append(f.path)
        elif hasattr(f, 'name'):
            paths.append(f.name)
    return [p for p in paths if p]


def process_videos(
    video_input, batch_files,
    yolo_model, conf_thresh, seq_len, pred_len, lstm_mode, class_filter,
    # Session
    session_notes,
    # Traffic analysis
    enable_traffic, counting_lines_json, ref_real_m, ref_px,
    enable_speed, enable_od,
    # Urban analysis
    enable_urban, enable_heatmap, heatmap_classes, zones_json, zone_areas_json,
    # Calibration
    calib_mode,
    # Export
    export_format, enable_mc_dropout, chart_format,
    progress=gr.Progress(track_tqdm=True),
):
    # ── 입력 경로 결정 ────────────────────────────────────────────
    batch_paths = _resolve_file_paths(batch_files)
    if batch_paths:
        paths = batch_paths
    elif video_input is not None:
        vp = video_input
        if isinstance(vp, dict):
            vdata = vp.get("video", {})
            vp = vdata.get("path") if isinstance(vdata, dict) else str(vdata)
        paths = [vp] if vp else []
    else:
        paths = []

    if not paths:
        return (None, None, None, "영상을 먼저 업로드해주세요.") + (None,) * 9

    is_batch = len(paths) > 1
    total_videos = len(paths)

    # ── 공통 설정 ─────────────────────────────────────────────────
    config = {**BASE_CONFIG,
              "yolo_model": yolo_model,
              "conf_thresh": conf_thresh,
              "seq_len": int(seq_len),
              "pred_len": int(pred_len)}

    pipeline = TrackingPipeline(config, lstm_device=LSTM_DEVICE, yolo_device=YOLO_DEVICE)
    if not pipeline.load_model(PRETRAINED_PATH):
        pretrain_model(config, PRETRAINED_PATH)
        pipeline.load_model(PRETRAINED_PATH)

    class_filter_set = set(class_filter) if class_filter else None
    mode = "finetune" if lstm_mode == "온라인 파인튜닝" else "pretrained"

    scale_mpp = None
    if ref_real_m and ref_px and ref_px > 0:
        scale_mpp = float(ref_real_m) / float(ref_px)

    counting_lines = _safe_json(counting_lines_json, []) if enable_traffic else []
    zones = _safe_json(zones_json, [])
    zone_areas = _safe_json(zone_areas_json, {})

    analysis_config = {
        'enable_traffic': bool(enable_traffic),
        'counting_lines': counting_lines,
        'scale_mpp': scale_mpp,
        'enable_speed': bool(enable_speed) and scale_mpp is not None,
        'enable_od': bool(enable_od),
        'enable_urban': bool(enable_urban),
        'enable_heatmap': bool(enable_heatmap),
        'heatmap_classes': list(heatmap_classes) if heatmap_classes else ['person'],
        'zones': zones,
        'zone_areas': zone_areas,
        'export_format': export_format,
    }

    # ── 영상 순차 처리 ────────────────────────────────────────────
    all_results = []
    for i, vpath in enumerate(paths):
        def _cb(pct, total, msg="", _i=i, _name=Path(vpath).name):
            overall = (_i + pct / 100) / total_videos
            progress(overall, desc=f"[{_i+1}/{total_videos}] {_name} — {msg}")

        result = pipeline.run(
            vpath,
            class_filter=class_filter_set,
            lstm_mode=mode,
            progress_callback=_cb,
            analysis_config=analysis_config,
            uncertainty=bool(enable_mc_dropout),
        )
        all_results.append((vpath, result))

    # ── 마지막 영상 기준 상세 출력 ────────────────────────────────
    last_vpath, last_result = all_results[-1]
    last_stats = last_result['stats']
    last_analysis = last_result.get('analysis', {})

    if is_batch:
        total_frames = sum(r['stats']['total_frames'] for _, r in all_results)
        total_tracks = sum(r['stats']['total_tracks'] for _, r in all_results)
        status = (
            f"완료  |  {total_videos}/{total_videos} 영상 처리됨"
            f"  |  총 프레임: {total_frames}"
            f"  |  총 추적 객체: {total_tracks}"
        )
    else:
        status = (
            f"완료  |  총 프레임: {last_stats['total_frames']}"
            f"  |  추적 객체 수: {last_stats['total_tracks']}"
            f"  |  장치: {DEVICE_DESC}"
        )

    # -- Traffic summary DataFrame --
    traffic_df = None
    counting_data = last_analysis.get('counting_lines', [])
    if counting_data:
        rows = []
        for line in counting_data:
            for cls, cnt in line.get('counts', {}).items():
                rows.append({
                    '감지선': line['label'],
                    '클래스': cls,
                    '통과 수': cnt,
                    '유량 (대/시)': line.get('flow_rates_veh_hr', {}).get(cls, ''),
                    '평균 차두시간 (s)': line.get('headway', {}).get(cls, {}).get('mean_s', ''),
                    '속도 P85 (km/h)': '',
                })
        if rows:
            traffic_df = pd.DataFrame(rows)

    speed_data = last_analysis.get('speed', {})
    if speed_data and speed_data.get('per_class'):
        spd_rows = [{'클래스': cls, **{f'속도_{k}': v for k, v in st.items()}}
                    for cls, st in speed_data['per_class'].items()]
        speed_df = pd.DataFrame(spd_rows)
    else:
        speed_df = None

    if traffic_df is not None and speed_df is not None:
        traffic_df = traffic_df.merge(
            speed_df[['클래스', '속도_p85']].rename(columns={'속도_p85': '속도 P85 (km/h)'}),
            on='클래스', how='left', suffixes=('', '_spd')
        )
        if '속도 P85 (km/h)_spd' in traffic_df.columns:
            traffic_df['속도 P85 (km/h)'] = traffic_df['속도 P85 (km/h)_spd']
            traffic_df.drop(columns=['속도 P85 (km/h)_spd'], inplace=True)
    elif speed_df is not None and traffic_df is None:
        traffic_df = speed_df

    od_df = None
    od_data = last_analysis.get('od_matrix', {})
    if od_data and 'matrix_df' in od_data:
        od_df = od_data['matrix_df'].reset_index()
        od_df.rename(columns={'index': 'Origin \\ Dest'}, inplace=True)

    zone_df = None
    zone_data = last_analysis.get('zone_analysis', {})
    if zone_data and zone_data.get('zone_summaries'):
        rows = []
        for zs in zone_data['zone_summaries']:
            util = zs.get('utilization') or {}
            rows.append({
                '존': zs['zone'],
                '입장 횟수': zs['entry_count'],
                '평균 체류 (s)': zs['mean_dwell_s'],
                '최대 체류 (s)': zs['max_dwell_s'],
                '평균 밀도 (명/m²)': util.get('mean_density_per_sqm', ''),
                '피크 밀도 (명/m²)': util.get('peak_density_per_sqm', ''),
            })
        if rows:
            zone_df = pd.DataFrame(rows)

    # -- Session (단일 모드만) --
    session_out = None
    if not is_batch:
        tmp_dir = Path(tempfile.mkdtemp())
        session_path = tmp_dir / "session.json"
        session_cfg = SessionConfig(
            video_hash=AnalysisSession.compute_video_hash(last_vpath),
            yolo_model=yolo_model,
            conf_thresh=conf_thresh,
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            lstm_mode=mode,
            class_filter=list(class_filter) if class_filter else [],
            counting_lines=counting_lines,
            zones=zones,
            zone_areas=zone_areas,
            scale_mpp=scale_mpp,
            enable_traffic=bool(enable_traffic),
            enable_speed=bool(enable_speed),
            enable_od=bool(enable_od),
            enable_urban=bool(enable_urban),
            enable_heatmap=bool(enable_heatmap),
            heatmap_classes=list(heatmap_classes) if heatmap_classes else [],
            export_format=export_format,
            enable_mc_dropout=bool(enable_mc_dropout),
            chart_format=chart_format,
            notes=session_notes or '',
            stats={'total_frames': last_stats['total_frames'],
                   'total_tracks': last_stats['total_tracks'],
                   'device': DEVICE_DESC},
        )
        AnalysisSession.save(session_path, session_cfg)
        session_out = str(session_path)

    # ── 배치 전용 출력 ────────────────────────────────────────────
    batch_summary_df = None
    batch_zip_path = None

    if is_batch:
        summary_rows = []
        for idx, (vp, res) in enumerate(all_results):
            summary_rows.append({
                '순번': idx + 1,
                '파일명': Path(vp).name,
                '총 프레임': res['stats']['total_frames'],
                '추적 객체 수': res['stats']['total_tracks'],
                '처리 시간 (s)': f"{res['duration_s']:.1f}",
                '상태': '완료',
            })
        batch_summary_df = pd.DataFrame(summary_rows)

        zip_tmp = Path(tempfile.mkdtemp()) / "batch_results.zip"
        with zipfile.ZipFile(zip_tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for idx, (vp, res) in enumerate(all_results):
                prefix = f"{idx+1:02d}_{Path(vp).stem}"
                for key, arcname in [
                    ('video', 'annotated.mp4'),
                    ('csv', 'predictions.csv'),
                    ('trajectory_img', 'trajectory.png'),
                    ('heatmap_img', 'heatmap.png'),
                    ('excel', 'report.xlsx'),
                    ('charts', 'charts.html'),
                ]:
                    fpath = res.get(key)
                    if fpath and Path(fpath).exists():
                        zf.write(fpath, f"{prefix}/{arcname}")
        batch_zip_path = str(zip_tmp)

    return (
        last_result['video'],
        last_result['csv'],
        last_result['trajectory_img'],
        status,
        traffic_df,
        od_df,
        last_result.get('heatmap_img'),
        zone_df,
        last_result.get('excel'),
        last_result.get('charts'),
        session_out,
        batch_summary_df,
        batch_zip_path,
    )


# ── CSS ───────────────────────────────────────────────────────────────────

css = """
/* ── Light mode page background ──────────────────────────── */
html, body {
    background: #f0f0f0 !important;
}
html.dark, html.dark body {
    background: #161616 !important;
}

/* ── Gradio container — full viewport width ───────────────── */
.gradio-container {
    background: transparent !important;
    max-width: 100% !important;
    padding-left: 28px !important;
    padding-right: 28px !important;
    box-sizing: border-box !important;
}

/* ── Run button ─────────────────────────────────────────── */
#run-btn button {
    background: #1d3a56 !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 13px 0 !important;
}
#run-btn button:hover { background: #274f78 !important; }
#run-btn button:active { background: #162c42 !important; }

/* ── Status box ──────────────────────────────────────────── */
#status-box textarea {
    font-size: 12px !important;
    font-family: 'Menlo', 'Consolas', 'Courier New', monospace !important;
}
"""

# ── Helper ────────────────────────────────────────────────────────────────

def _sec(text):
    """Section label with inline styles — works regardless of Gradio CSS scoping."""
    return gr.HTML(
        f'<p style="'
        f'font-size:14px;font-weight:700;letter-spacing:0.09em;'
        f'text-transform:uppercase;'
        f'color:var(--body-text-color);'
        f'padding:0 0 0 10px;'
        f'border-left:3px solid var(--border-color-accent);'
        f'margin:24px 0 8px 0;line-height:1.5;display:block;'
        f'">{text}</p>'
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Object Tracking + Trajectory Prediction",
               theme=gr.themes.Monochrome(), css=css) as demo:

    gr.HTML(
        '<div style="text-align:center;border-bottom:1px solid var(--border-color-primary);'
        'padding-bottom:14px;margin-bottom:8px;">'
        '<h1 style="font-size:18px;font-weight:700;color:var(--body-text-color);'
        'letter-spacing:0.03em;margin:0 0 6px 0;">'
        'Object Tracking + Trajectory Prediction'
        '</h1>'
        '<p style="font-size:12px;color:var(--body-text-color-subdued);margin:0;">'
        'Multi-Object Tracking (ByteTrack) &nbsp;&middot;&nbsp; '
        'LSTM Trajectory Prediction &nbsp;&middot;&nbsp; Research Analysis Tool'
        '</p>'
        '</div>'
    )

    with gr.Row():
        # ── Left panel ──────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=360):

            _sec("세션")
            with gr.Group():
                session_notes = gr.Textbox(
                    label="연구자 메모",
                    placeholder="연구 목적, 촬영 조건, 카메라 위치 등...",
                    lines=2,
                )
                session_load = gr.File(
                    label="세션 불러오기 (.json)",
                    file_types=[".json"],
                )

            _sec("영상 업로드")
            with gr.Tabs():
                with gr.Tab("단일 영상"):
                    video_input = gr.Video(label="영상 파일  (MP4 / AVI / MOV)")
                with gr.Tab("배치 처리"):
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                        label="영상 파일 목록  (여러 파일 선택 가능)",
                    )

            _sec("기본 설정")
            with gr.Group():
                yolo_model = gr.Dropdown(
                    choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
                             "yolo11l.pt", "yolo11x.pt"],
                    value="yolo11n.pt",
                    label="YOLO 모델",
                    info="n = 빠름 / x = 정확",
                )
                conf_thresh = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.05,
                    label="감지 신뢰도 임계값",
                )
                seq_len = gr.Slider(
                    5, 50, value=20, step=1,
                    label="과거 시퀀스 길이 (프레임)",
                )
                pred_len = gr.Slider(
                    1, 30, value=10, step=1,
                    label="예측 프레임 수",
                )
                lstm_mode = gr.Radio(
                    choices=["사전학습 모델", "온라인 파인튜닝"],
                    value="사전학습 모델",
                    label="LSTM 모드",
                    info="파인튜닝: 영상 앞 30%로 모델 조정 후 예측",
                )
                class_filter = gr.CheckboxGroup(
                    choices=COCO_TRACKABLE,
                    value=["person", "car"],
                    label="추적 대상 클래스",
                )

            # Traffic Analysis
            with gr.Accordion("교통 분석", open=False):
                enable_traffic = gr.Checkbox(label="교통 분석 활성화", value=False)
                gr.Markdown(
                    "감지선 정의 (JSON) — 가상 루프 검지기처럼 통과 차량을 계수합니다.",
                    elem_classes="note",
                )
                counting_lines_json = gr.Textbox(
                    value=DEFAULT_LINES,
                    label='감지선  (JSON: label, x1, y1, x2, y2)',
                    lines=3,
                )
                gr.Markdown(
                    "속도 추정 스케일 — 실거리와 픽셀 거리를 입력하면 km/h로 변환합니다.",
                    elem_classes="note",
                )
                with gr.Row():
                    ref_real_m = gr.Number(label="실거리 (m)", value=3.5,
                                           info="예: 차선폭 3.5 m")
                    ref_px = gr.Number(label="픽셀 거리 (px)", value=100,
                                       info="해당 거리의 픽셀 수")
                with gr.Row():
                    enable_speed = gr.Checkbox(label="속도 추정 (km/h)", value=False)
                    enable_od = gr.Checkbox(label="OD 행렬", value=False)

            # Urban Analysis
            with gr.Accordion("도시 공간 분석", open=False):
                enable_urban = gr.Checkbox(label="도시 분석 활성화", value=False)
                enable_heatmap = gr.Checkbox(label="밀도 열지도 생성", value=True)
                heatmap_classes = gr.CheckboxGroup(
                    choices=COCO_TRACKABLE,
                    value=["person"],
                    label="열지도 대상 클래스",
                )
                gr.Markdown(
                    "존 정의 (JSON) — 교통 분석의 OD 행렬과 공유됩니다.",
                    elem_classes="note",
                )
                zones_json = gr.Textbox(
                    value=DEFAULT_ZONES,
                    label='존 정의  (JSON: name, polygon [[x,y],...])',
                    lines=4,
                    placeholder='[{"name": "Zone A", "polygon": [[100,100],[300,100],[300,300],[100,300]]}]',
                )
                zone_areas_json = gr.Textbox(
                    value=DEFAULT_ZONE_AREAS,
                    label='존 면적  (JSON, m²) — 밀도 계산용',
                    lines=2,
                    placeholder='{"Zone A": 50.0, "Zone B": 30.0}',
                )

            # Calibration
            with gr.Accordion("캘리브레이션", open=False):
                calib_mode = gr.Radio(
                    choices=["없음", "기준거리"],
                    value="없음",
                    label="캘리브레이션 모드",
                    info="'기준거리': 위의 실거리·픽셀거리로 m/px 자동 계산",
                )

            _sec("내보내기")
            with gr.Group():
                export_format = gr.Radio(
                    choices=["CSV만", "CSV + Excel", "CSV + Excel + 차트"],
                    value="CSV만",
                    label="내보내기 형식",
                )
                enable_mc_dropout = gr.Checkbox(
                    label="LSTM 불확실성 추정 (MC-Dropout — 처리 느려짐)",
                    value=False,
                )
                chart_format = gr.Radio(
                    choices=["HTML (인터랙티브)", "SVG (인쇄용)"],
                    value="HTML (인터랙티브)",
                    label="차트 형식",
                )

        # ── Right panel ──────────────────────────────────────────────────
        with gr.Column(scale=3):
            run_btn = gr.Button("처리 시작", variant="primary",
                                size="lg", elem_id="run-btn")
            status_box = gr.Textbox(label="상태", interactive=False,
                                    lines=1, elem_id="status-box")
            monitor_summary = gr.Textbox(
                label="시스템",
                interactive=False,
                lines=1,
                elem_id="monitor-summary",
            )
            with gr.Accordion("상세 모니터링", open=False):
                with gr.Row():
                    cpu_plot = gr.Plot(label="CPU 사용률")
                    ram_plot = gr.Plot(label="RAM 사용률")
                    gpu_plot = gr.Plot(label="GPU 사용률")
            monitor_timer = gr.Timer(value=2.0, active=True)

            gr.HTML(
                '<p style="font-size:14px;font-weight:700;letter-spacing:0.09em;'
                'text-transform:uppercase;color:var(--body-text-color);'
                'padding:0 0 0 10px;border-left:3px solid var(--border-color-accent);'
                'margin:16px 0 8px 0;line-height:1.5;display:block;">추적 결과</p>'
            )
            with gr.Row():
                video_output = gr.Video(label="어노테이션 영상")
                summary_img = gr.Image(label="궤적 요약", type="filepath")

            with gr.Accordion("교통 분석 결과", open=False):
                gr.Markdown(
                    "감지선 통과 통계 — 클래스별 계수·유량·차두시간·속도",
                    elem_classes="note",
                )
                traffic_df_out = gr.DataFrame(label="교통 지표")
                gr.Markdown("Origin-Destination 행렬", elem_classes="note")
                od_df_out = gr.DataFrame(label="OD 행렬")

            with gr.Accordion("도시 분석 결과", open=False):
                heatmap_img_out = gr.Image(label="밀도 열지도", type="filepath")
                zone_df_out = gr.DataFrame(label="존 분석  (체류시간 · 밀도)")

            gr.HTML(
                '<p style="font-size:14px;font-weight:700;letter-spacing:0.09em;'
                'text-transform:uppercase;color:var(--body-text-color);'
                'padding:0 0 0 10px;border-left:3px solid var(--border-color-accent);'
                'margin:24px 0 8px 0;line-height:1.5;display:block;">내보내기</p>'
            )
            with gr.Row():
                csv_download = gr.File(label="예측 데이터  (CSV)")
                excel_download = gr.File(label="연구 보고서  (Excel)")
            with gr.Row():
                charts_download = gr.File(label="분석 차트  (HTML / SVG)")
                session_download = gr.File(label="세션 파일  (JSON)")

            gr.HTML(
                '<p style="font-size:14px;font-weight:700;letter-spacing:0.09em;'
                'text-transform:uppercase;color:var(--body-text-color);'
                'padding:0 0 0 10px;border-left:3px solid var(--border-color-accent);'
                'margin:24px 0 8px 0;line-height:1.5;display:block;">배치 처리 결과</p>'
            )
            batch_summary_df_out = gr.DataFrame(label="영상별 처리 요약")
            batch_zip_download = gr.File(label="전체 결과 다운로드 (ZIP)")

    _all_outputs = [
        video_output, csv_download, summary_img, status_box,
        traffic_df_out, od_df_out,
        heatmap_img_out, zone_df_out,
        excel_download, charts_download, session_download,
        batch_summary_df_out, batch_zip_download,
    ]

    run_btn.click(
        fn=process_videos,
        inputs=[
            video_input, batch_files,
            yolo_model, conf_thresh, seq_len, pred_len,
            lstm_mode, class_filter,
            session_notes,
            enable_traffic, counting_lines_json, ref_real_m, ref_px,
            enable_speed, enable_od,
            enable_urban, enable_heatmap, heatmap_classes,
            zones_json, zone_areas_json,
            calib_mode,
            export_format, enable_mc_dropout, chart_format,
        ],
        outputs=_all_outputs,
    )

    video_input.clear(
        fn=lambda: (None,) * 13,
        inputs=[],
        outputs=_all_outputs,
    )

    monitor_timer.tick(
        fn=_update_monitor,
        inputs=[],
        outputs=[monitor_summary, cpu_plot, ram_plot, gpu_plot],
    )

    gr.HTML(
        '<p style="font-size:11px;color:var(--body-text-color-subdued);'
        'border-top:1px solid var(--border-color-primary);'
        'margin-top:14px;padding-top:10px;line-height:1.9;">'
        '실선 = 과거 이동 경로 &nbsp;|&nbsp; 점선 = 예측 궤적 &nbsp;|&nbsp;'
        ' 청색선 = 감지선 &nbsp;|&nbsp; 오렌지 폴리곤 = 분석 존 &nbsp;|&nbsp;'
        ' 각 색상은 고유 Track ID를 나타냅니다.'
        '</p>'
    )

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware

    class _SABHeadersMiddleware(BaseHTTPMiddleware):
        """Add COOP/COEP headers so FFmpeg.wasm (Gradio VideoEditor trim) can use SharedArrayBuffer."""
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            return response

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(_SABHeadersMiddleware)
    gr.mount_gradio_app(fastapi_app, demo, path="/")

    print("[서버] http://127.0.0.1:7860")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=7860)
