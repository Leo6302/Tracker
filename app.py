import sys
import json
import copy
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import torch
import yaml
import gradio as gr

from src.pipeline import TrackingPipeline
from src.prediction.trainer import pretrain_model
from src.device_utils import detect_devices
from src.analysis.session import AnalysisSession, SessionConfig
from src.analysis.ai_insight import (
    build_analysis_summary, generate_insight, render_insight_markdown, AIInsightError,
    DEFAULT_MODEL, MODEL_CHOICES, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_HOST,
)

import psutil
from collections import deque
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import GPUtil as _gputil_mod
    _GPUTIL_AVAILABLE = True
except Exception:
    _gputil_mod = None
    _GPUTIL_AVAILABLE = False

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PRETRAINED_PATH = Path(__file__).parent / "models" / "lstm_pretrained.pt"
GUIDE_IMG_DIR = Path(__file__).parent / "assets" / "guide"

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

_STATS_HISTORY: deque = deque(maxlen=60)
psutil.cpu_percent(interval=None)


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
        'cpu': cpu, 'ram': mem.percent, 'ram_gb': mem.used / 1e9,
        'gpu': gpu_load, 'gpu_mem': gpu_mem, 'has_gpu': has_gpu,
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


def _make_traffic_count_chart(counting_data):
    """Grouped bar chart (Plotly): per-class count and flow rate per counting line."""
    if not counting_data:
        return None
    all_cls = sorted({cls for line in counting_data for cls in line.get('counts', {})})
    if not all_cls:
        return None

    n_lines = len(counting_data)
    colors = px.colors.qualitative.Set1
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['클래스별 통과 수', '클래스별 유량 (대/시)'],
        horizontal_spacing=0.12,
    )

    for i, line in enumerate(counting_data):
        lbl = line.get('label', f'Line {i+1}')
        counts = [line.get('counts', {}).get(cls, 0) for cls in all_cls]
        flows = [line.get('flow_rates_veh_hr', {}).get(cls, 0) for cls in all_cls]
        color = colors[i % len(colors)]

        fig.add_trace(go.Bar(
            name=lbl, x=all_cls, y=counts,
            marker_color=color, legendgroup=lbl,
            hovertemplate='<b>%{x}</b><br>통과 수: <b>%{y}</b><extra>' + lbl + '</extra>',
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=lbl, x=all_cls, y=[round(v, 1) for v in flows],
            marker_color=color, legendgroup=lbl, showlegend=False,
            hovertemplate='<b>%{x}</b><br>유량: <b>%{y:.1f}</b> 대/시<extra>' + lbl + '</extra>',
        ), row=1, col=2)

    fig.update_layout(
        barmode='group',
        height=420,
        legend_title_text='감지선',
        yaxis_title='통과 수',
        yaxis2_title='유량 (대/시)',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(t=60, b=50, l=60, r=20),
    )
    fig.update_xaxes(tickangle=-20)
    return fig


def _make_speed_chart(speed_data):
    """Box plot (Plotly): speed distribution per class using raw track samples."""
    if not speed_data:
        return None
    track_speeds = speed_data.get('track_speeds', {})
    track_cls = speed_data.get('track_cls', {})
    if not track_speeds:
        return None

    by_class: dict = {}
    for tid, speeds in track_speeds.items():
        cls = track_cls.get(tid) or track_cls.get(str(tid), 'unknown')
        by_class.setdefault(cls, []).extend(speeds)

    labels = [cls for cls in sorted(by_class) if len(by_class[cls]) >= 2]
    if not labels:
        return None

    colors = px.colors.qualitative.Set1
    fig = go.Figure()
    for i, cls in enumerate(labels):
        color = colors[i % len(colors)]
        fig.add_trace(go.Box(
            y=by_class[cls],
            name=cls,
            marker_color=color,
            boxpoints='outliers',
            marker_size=4,
            line_width=2,
            hovertemplate='속도: <b>%{y:.1f}</b> km/h<extra>' + cls + '</extra>',
        ))

    fig.update_layout(
        title='클래스별 속도 분포',
        yaxis_title='속도 (km/h)',
        height=420,
        showlegend=False,
        template='plotly_white',
        hovermode='closest',
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def _annotate_chart(fig, highlights, category, xref="x", yref="y domain"):
    """AI 하이라이트 중 category에 해당하는 항목을 fig 위에 작은 배지로 표시한다.

    원본 fig는 건드리지 않고 깊은 복사본에 주석을 추가해 반환 — 재생성할 때마다
    이전 주석이 누적되지 않도록 항상 process_videos()가 만든 원본에서 다시 그린다.
    """
    if fig is None or not highlights:
        return fig
    targets = [h for h in highlights if h.get("category") == category]
    if not targets:
        return fig
    fig = copy.deepcopy(fig)
    for h in targets:
        color = "#b45309" if h.get("importance") == "high" else "#1d4ed8"
        fig.add_annotation(
            x=h.get("target"), y=1.08, xref=xref, yref=yref,
            text=f"💡 {h.get('note', '')}", showarrow=False,
            font=dict(size=10, color="#fff"), bgcolor=color,
            borderpad=4, xanchor="center",
        )
    return fig


def _highlight_dataframe(df, highlights, match_column, category):
    """AI 하이라이트 중 category에 해당하는 항목을 df에 "AI 메모" 컬럼으로 덧붙이고,
    일치하는 행에 배경색을 입힌 pandas Styler를 반환한다.

    match_column 값이 하이라이트의 target과 일치하지 않으면 그 행은 그대로 둔다.
    """
    if df is None or df.empty:
        return df
    targets = {h["target"]: h for h in highlights if h.get("category") == category}
    if not targets:
        return df

    df = df.copy()
    df["AI 메모"] = df[match_column].map(lambda v: targets.get(v, {}).get("note", ""))

    def _row_style(row):
        h = targets.get(row[match_column])
        if not h:
            return [""] * len(row)
        bg = "#fef3c7" if h.get("importance") == "high" else "#eff6ff"
        # 다크 테마에서도 글자가 보이도록 글자색을 명시적으로 어둡게 고정
        # (배경만 지정하면 테마의 기본(밝은) 글자색과 겹쳐 안 보이는 문제가 있었음)
        return [f"background-color: {bg}; color: #1a1a1a"] * len(row)

    return df.style.apply(_row_style, axis=1)


def _build_video_package(vpath, result, zones, enable_od, enabled_flags, label):
    """단일 영상의 처리 결과로부터 화면에 표시할 차트·표·AI 인사이트 스냅샷을 만든다.

    배치 처리 시 영상마다 한 번씩 호출되어, 영상별 패키지를 독립적으로 보관함으로써
    나중에 사용자가 드롭다운으로 영상을 골라 다시 확인할 수 있게 한다.
    """
    analysis = result.get('analysis', {})

    counting_data = analysis.get('counting_lines', [])
    speed_data = analysis.get('speed', {})
    traffic_count_fig = _make_traffic_count_chart(counting_data)
    traffic_speed_fig = _make_speed_chart(speed_data)

    od_df = None
    od_data = analysis.get('od_matrix', {})
    if od_data and 'matrix_df' in od_data:
        od_df = od_data['matrix_df'].reset_index()
        od_df.rename(columns={'index': 'Origin \\ Dest'}, inplace=True)
    elif enable_od:
        if len(zones) < 2:
            od_msg = (
                "OD 행렬에는 최소 2개 이상의 존이 필요합니다. "
                "'도시 공간 분석' 아코디언에서 존을 2개 이상 활성화하고 좌표를 입력해주세요."
            )
        else:
            od_msg = "이번 영상에서는 활성화된 존 사이를 이동한 객체가 감지되지 않았습니다."
        od_df = pd.DataFrame({"안내": [od_msg]})

    zone_df = None
    zone_data = analysis.get('zone_analysis', {})
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

    track_summary_df = pd.DataFrame(result.get('track_summary', []))
    ai_summary = build_analysis_summary(analysis, result, enabled_flags)
    ai_insight_snapshot = {
        "llm_summary": ai_summary,
        "traffic_count_fig": traffic_count_fig,
        "traffic_speed_fig": traffic_speed_fig,
        "zone_df": zone_df,
        "track_summary_df": track_summary_df,
    }

    return {
        "label": label,
        "video": result['video'],
        "csv": result['csv'],
        "trajectory_img": result['trajectory_img'],
        "heatmap_img": result.get('heatmap_img'),
        "excel": result.get('excel'),
        "charts": result.get('charts'),
        "traffic_count_fig": traffic_count_fig,
        "traffic_speed_fig": traffic_speed_fig,
        "od_df": od_df,
        "zone_df": zone_df,
        "track_summary_df": track_summary_df,
        "ai_insight_snapshot": ai_insight_snapshot,
    }


# ── process_videos ────────────────────────────────────────────────────────
# Line inputs: 4 lines × (enable, label, x1, y1, x2, y2) = 24 params
# Zone inputs: 4 zones × (enable, name, x1, y1, x2, y2, area) = 28 params

def process_videos(
    video_input, batch_files,
    yolo_model, conf_thresh, seq_len, pred_len, lstm_mode, class_filter,
    session_notes,
    enable_traffic,
    l1_en, l1_lbl, l1_x1, l1_y1, l1_x2, l1_y2,
    l2_en, l2_lbl, l2_x1, l2_y1, l2_x2, l2_y2,
    l3_en, l3_lbl, l3_x1, l3_y1, l3_x2, l3_y2,
    l4_en, l4_lbl, l4_x1, l4_y1, l4_x2, l4_y2,
    ref_real_m, ref_px, enable_speed, enable_od,
    enable_urban, enable_heatmap, heatmap_classes,
    z1_en, z1_nm, z1_x1, z1_y1, z1_x2, z1_y2, z1_area,
    z2_en, z2_nm, z2_x1, z2_y1, z2_x2, z2_y2, z2_area,
    z3_en, z3_nm, z3_x1, z3_y1, z3_x2, z3_y2, z3_area,
    z4_en, z4_nm, z4_x1, z4_y1, z4_x2, z4_y2, z4_area,
    calib_mode,
    export_format, enable_mc_dropout, chart_format,
    progress=gr.Progress(track_tqdm=True),
):
    # ── 입력 경로 결정 ────────────────────────────────────────────
    batch_paths = _resolve_file_paths(batch_files)
    if batch_paths:
        paths = batch_paths
    elif video_input is not None:
        paths = [str(video_input)]
    else:
        paths = []

    if not paths:
        return (
            (None, None, None, "영상을 먼저 업로드해주세요.") + (None,) * 13
            + (gr.update(choices=[], value=None),)
        )

    is_batch = len(paths) > 1
    total_videos = len(paths)

    # ── 감지선 JSON 빌드 ──────────────────────────────────────────
    _line_raw = [
        (l1_en, l1_lbl, l1_x1, l1_y1, l1_x2, l1_y2),
        (l2_en, l2_lbl, l2_x1, l2_y1, l2_x2, l2_y2),
        (l3_en, l3_lbl, l3_x1, l3_y1, l3_x2, l3_y2),
        (l4_en, l4_lbl, l4_x1, l4_y1, l4_x2, l4_y2),
    ]
    counting_lines = [
        {
            "label": (lbl or f"Line {i+1}"),
            "x1": int(x1 or 0), "y1": int(y1 or 0),
            "x2": int(x2 or 0), "y2": int(y2 or 0),
        }
        for i, (en, lbl, x1, y1, x2, y2) in enumerate(_line_raw) if en
    ] if enable_traffic else []

    # ── 존 JSON 빌드 ──────────────────────────────────────────────
    _zone_raw = [
        (z1_en, z1_nm, z1_x1, z1_y1, z1_x2, z1_y2, z1_area),
        (z2_en, z2_nm, z2_x1, z2_y1, z2_x2, z2_y2, z2_area),
        (z3_en, z3_nm, z3_x1, z3_y1, z3_x2, z3_y2, z3_area),
        (z4_en, z4_nm, z4_x1, z4_y1, z4_x2, z4_y2, z4_area),
    ]
    zones = []
    zone_areas = {}
    for i, (en, nm, x1, y1, x2, y2, area) in enumerate(_zone_raw):
        if not en:
            continue
        nm = nm or f"Zone {chr(65 + i)}"
        x1, y1, x2, y2 = int(x1 or 0), int(y1 or 0), int(x2 or 0), int(y2 or 0)
        zones.append({"name": nm, "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
        if area:
            zone_areas[nm] = float(area)

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

    last_vpath, last_result = all_results[-1]
    last_stats = last_result['stats']

    total_frames = sum(r['stats']['total_frames'] for _, r in all_results)
    total_tracks = sum(r['stats']['total_tracks'] for _, r in all_results)

    if is_batch:
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

    # ── 영상별 표시 패키지 (차트·표·AI 인사이트 스냅샷) ────────────────────
    # 배치 처리 시 마지막 영상만 보여주던 것을 고쳐, 영상마다 패키지를 만들어
    # 보관해두고 드롭다운으로 골라 다시 확인할 수 있게 한다.
    enabled_flags = {
        "enable_traffic": bool(enable_traffic),
        "enable_speed": bool(enable_speed) and scale_mpp is not None,
        "enable_od": bool(enable_od),
        "enable_urban": bool(enable_urban),
    }
    packages = [
        _build_video_package(
            vp, res, zones, bool(enable_od), enabled_flags,
            label=f"{idx+1:02d}. {Path(vp).name}",
        )
        for idx, (vp, res) in enumerate(all_results)
    ]
    last_pkg = packages[-1]

    # ── 세션 저장 (단일/배치 공통) ───────────────────────────────────
    notes_text = session_notes or ''
    if is_batch:
        file_list = ', '.join(Path(vp).name for vp, _ in all_results)
        batch_tag = f"[배치 처리 — {total_videos}개 영상: {file_list}]"
        notes_text = f"{notes_text}\n{batch_tag}" if notes_text else batch_tag

    tmp_dir = Path(tempfile.mkdtemp())
    session_path = tmp_dir / "session.json"
    session_cfg = SessionConfig(
        # 배치는 영상이 여럿이라 단일 video_hash로 대표할 수 없으므로 비워둠
        video_hash=('' if is_batch else AnalysisSession.compute_video_hash(last_vpath)),
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
        notes=notes_text,
        stats={'total_frames': total_frames,
               'total_tracks': total_tracks,
               'device': DEVICE_DESC},
    )
    AnalysisSession.save(session_path, session_cfg)
    session_out = str(session_path)

    # ── 배치 전용 출력 ────────────────────────────────────────────
    batch_summary_df = None
    batch_zip_path = None
    batch_video_choices = gr.update(choices=[], value=None)

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

        batch_video_choices = gr.update(
            choices=[pkg['label'] for pkg in packages],
            value=last_pkg['label'],
        )

    # 18 outputs
    return (
        last_pkg['video'],
        last_pkg['csv'],
        last_pkg['trajectory_img'],
        status,
        last_pkg['traffic_count_fig'],
        last_pkg['traffic_speed_fig'],
        last_pkg['od_df'],
        last_pkg['heatmap_img'],
        last_pkg['zone_df'],
        last_pkg['track_summary_df'],
        last_pkg['excel'],
        last_pkg['charts'],
        session_out,
        batch_summary_df,
        batch_zip_path,
        last_pkg['ai_insight_snapshot'],
        packages,
        batch_video_choices,
    )


def restore_session(session_file):
    # 68 outputs: 4 + 3 + 1 + 24(lines) + 2 + 3 + 28(zones) + 3
    if not session_file:
        return (gr.skip(),) * 68
    path = session_file if isinstance(session_file, str) else session_file.name
    cfg = AnalysisSession.load(path)
    gr.Info(f"세션 불러오기 완료  (생성일: {cfg.created_at or '알 수 없음'})")
    lstm_mode_label = "온라인 파인튜닝" if cfg.lstm_mode == "finetune" else "사전학습 모델"

    # Restore 4 counting line slots
    lines = cfg.counting_lines or []
    line_vals = [(False, f"Line {i+1}", 0, 0, 0, 0) for i in range(4)]
    for i, ln in enumerate(lines[:4]):
        line_vals[i] = (
            True, ln.get('label', f'Line {i+1}'),
            int(ln.get('x1', 0)), int(ln.get('y1', 0)),
            int(ln.get('x2', 0)), int(ln.get('y2', 0)),
        )
    line_flat = [v for t in line_vals for v in t]

    # Restore 4 zone slots
    zones_list = cfg.zones or []
    zone_areas_dict = cfg.zone_areas or {}
    zone_vals = [(False, f"Zone {chr(65+i)}", 0, 0, 0, 0, 0.0) for i in range(4)]
    for i, zone in enumerate(zones_list[:4]):
        nm = zone.get('name', f'Zone {chr(65+i)}')
        poly = zone.get('polygon', [[0, 0], [0, 0], [0, 0], [0, 0]])
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        zone_vals[i] = (True, nm, min(xs), min(ys), max(xs), max(ys),
                        float(zone_areas_dict.get(nm, 0.0)))
    zone_flat = [v for t in zone_vals for v in t]

    return (
        cfg.yolo_model, cfg.conf_thresh, cfg.seq_len, cfg.pred_len,
        lstm_mode_label, cfg.class_filter, cfg.notes or '',
        cfg.enable_traffic,
        *line_flat,
        cfg.enable_speed, cfg.enable_od,
        cfg.enable_urban, cfg.enable_heatmap, cfg.heatmap_classes,
        *zone_flat,
        cfg.export_format, cfg.enable_mc_dropout, cfg.chart_format,
    )


def run_ai_insight(snapshot, provider, claude_model, claude_api_key, ollama_model, ollama_host):
    if not snapshot:
        gr.Warning("먼저 영상을 처리해주세요. 분석 결과가 없습니다.")
        return (gr.skip(),) * 5

    provider_key = "ollama" if provider == "로컬 (Ollama)" else "claude"
    try:
        result = generate_insight(
            snapshot["llm_summary"], provider=provider_key,
            model=claude_model, api_key=(claude_api_key or None),
            ollama_model=ollama_model, ollama_host=ollama_host,
        )
    except AIInsightError as e:
        gr.Warning(str(e))
        return (gr.skip(),) * 5

    highlights = result.get("highlights") or []
    markdown = render_insight_markdown(result)
    count_fig = _annotate_chart(snapshot["traffic_count_fig"], highlights, "counting_lines")
    speed_fig = _annotate_chart(snapshot["traffic_speed_fig"], highlights, "speed")
    zone_styled = _highlight_dataframe(snapshot["zone_df"], highlights, "존", "zone_analysis")
    track_styled = _highlight_dataframe(snapshot["track_summary_df"], highlights, "class", "track_summary")
    return markdown, count_fig, speed_fig, zone_styled, track_styled


def view_batch_video(packages, label):
    """배치 처리 결과 중 사용자가 드롭다운에서 고른 영상의 패키지를 화면에 표시한다."""
    if not packages or not label:
        gr.Warning("선택할 영상이 없습니다. 먼저 배치 처리를 완료해주세요.")
        return (gr.skip(),) * 14
    pkg = next((p for p in packages if p["label"] == label), None)
    if pkg is None:
        gr.Warning("선택한 영상의 결과를 찾을 수 없습니다.")
        return (gr.skip(),) * 14
    return (
        pkg["video"], pkg["csv"], pkg["trajectory_img"],
        f"{label}  결과 표시 중",
        pkg["traffic_count_fig"], pkg["traffic_speed_fig"], pkg["od_df"],
        pkg["heatmap_img"], pkg["zone_df"], pkg["track_summary_df"],
        pkg["excel"], pkg["charts"],
        pkg["ai_insight_snapshot"], "",
    )


# ── CSS ───────────────────────────────────────────────────────────────────

css = """
html, body { background: #f0f0f0 !important; }
html.dark, html.dark body { background: #161616 !important; }
.gradio-container {
    background: transparent !important;
    max-width: 100% !important;
    padding-left: 28px !important;
    padding-right: 28px !important;
    box-sizing: border-box !important;
}
#run-btn {
    background: #1d3a56 !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 13px 0 !important;
}
#run-btn:hover { background: #274f78 !important; }
#run-btn:active { background: #162c42 !important; }
#status-box textarea {
    font-size: 12px !important;
    font-family: 'Menlo', 'Consolas', 'Courier New', monospace !important;
}

/* Modal overlays (settings / guide) — :not(.hide) so Gradio's own
   `.hide { display: none }` (added when visible=False) isn't fought by our
   `display: flex` override; an ID selector would otherwise outrank it. */
#settings-modal:not(.hide), #guide-modal:not(.hide) {
    position: fixed !important;
    inset: 0 !important;
    z-index: 1000 !important;
    background: rgba(0,0,0,0.5) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 24px !important;
}
#settings-panel, #guide-panel {
    background: var(--background-fill-primary) !important;
    border-radius: 6px !important;
    max-width: 880px !important;
    width: 100% !important;
    max-height: 86vh !important;
    overflow-y: auto !important;
    padding: 24px 28px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35) !important;
}
/* Gradio gives many block-level children (note Markdown, Group, collapsed
   Accordion) their own `overflow: auto` by default. Once the panel itself
   is a scrollable flex column, that per-child overflow triggers flexbox's
   "automatic minimum size" rule (a flex item with overflow != visible gets
   an effective min-height of 0), so those children get crushed toward 0px
   instead of the panel growing to its full content height and scrolling.
   Resetting descendants back to overflow:visible removes that trigger —
   nothing inside these panels needs its own internal scrollbar, only the
   panel itself does. */
#settings-panel *, #guide-panel * {
    overflow: visible !important;
}

/* Home screen — minimal, centered */
#home-screen {
    max-width: 640px !important;
    margin: 40px auto !important;
}
#home-screen #status-box textarea,
#home-screen #monitor-summary textarea {
    font-size: 11px !important;
    opacity: 0.7 !important;
}

/* Small circular "?" guide buttons beside results section headers */
.guide-btn {
    min-width: 28px !important;
    width: 28px !important;
    height: 28px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    background: transparent !important;
    border: 1px solid var(--border-color-primary) !important;
    color: var(--body-text-color-subdued) !important;
}
.guide-btn:hover { background: var(--background-fill-secondary) !important; }
"""


def _sec(text):
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

with gr.Blocks(title="Object Tracking + Trajectory Prediction") as demo:

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

    with gr.Column(elem_id="home-screen") as home_screen:
        _sec("영상 업로드")
        gr.Markdown(
            "처리할 동영상 파일을 선택합니다. 단일 영상 또는 배치 처리(여러 파일)를 선택할 수 있습니다.",
            elem_classes="note",
        )
        with gr.Tabs():
            with gr.Tab("단일 영상"):
                video_input = gr.File(
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    label="영상 파일  (MP4 / AVI / MOV)",
                )
            with gr.Tab("배치 처리"):
                batch_files = gr.File(
                    file_count="multiple",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    label="영상 파일 목록  (여러 파일 선택 가능)",
                )

        with gr.Row():
            open_settings_btn = gr.Button("세부설정")
            open_guide_btn = gr.Button("사용법 가이드")

        run_btn = gr.Button("처리 시작", variant="primary",
                            size="lg", elem_id="run-btn")
        status_box = gr.Textbox(label="상태", interactive=False,
                                lines=1, elem_id="status-box")
        monitor_summary = gr.Textbox(
            label="시스템", interactive=False, lines=1,
            elem_id="monitor-summary",
        )
        with gr.Accordion("상세 모니터링", open=False):
            with gr.Row():
                cpu_plot = gr.Plot(label="CPU 사용률")
                ram_plot = gr.Plot(label="RAM 사용률")
                gpu_plot = gr.Plot(label="GPU 사용률")
        monitor_timer = gr.Timer(value=2.0, active=True)

    with gr.Column(elem_id="results-screen", visible=False) as results_screen:
        new_analysis_btn = gr.Button("새 영상 분석", variant="primary")

        with gr.Row():
            gr.HTML(
                '<p style="font-size:14px;font-weight:700;letter-spacing:0.09em;'
                'text-transform:uppercase;color:var(--body-text-color);'
                'padding:0 0 0 10px;border-left:3px solid var(--border-color-accent);'
                'margin:16px 0 8px 0;line-height:1.5;display:block;">추적 결과</p>'
            )
            guide_btn_tracking = gr.Button("?", elem_classes="guide-btn", scale=0, min_width=36)
        with gr.Row():
            video_output = gr.Video(label="어노테이션 영상")
            summary_img  = gr.Image(label="궤적 요약", type="filepath")

        with gr.Row():
            batch_video_select = gr.Dropdown(
                label="결과를 확인할 영상 선택", choices=[], value=None, scale=3,
            )
            batch_view_btn = gr.Button("선택한 영상 결과 보기", scale=1, variant="primary")
        batch_packages_state = gr.State(value=None)

        with gr.Row():
            guide_btn_traffic = gr.Button("?", elem_classes="guide-btn", scale=0, min_width=36)
        with gr.Accordion("교통 분석 결과", open=False):
            traffic_count_plot = gr.Plot(label="통과 수 / 유량")
            traffic_speed_plot = gr.Plot(label="속도 분포")
            od_df_out = gr.DataFrame(label="OD 행렬")

        with gr.Row():
            guide_btn_urban = gr.Button("?", elem_classes="guide-btn", scale=0, min_width=36)
        with gr.Accordion("도시 분석 결과", open=False):
            heatmap_img_out = gr.Image(label="밀도 열지도", type="filepath")
            zone_df_out = gr.DataFrame(label="존 분석  (체류시간 · 밀도)", interactive=False)

        with gr.Row():
            guide_btn_track_summary = gr.Button("?", elem_classes="guide-btn", scale=0, min_width=36)
        with gr.Accordion("트랙 요약", open=True):
            track_summary_df_out = gr.DataFrame(label="트랙별 요약", interactive=False)

        with gr.Row():
            guide_btn_ai = gr.Button("?", elem_classes="guide-btn", scale=0, min_width=36)
        with gr.Accordion("AI 인사이트", open=True):
            ai_provider_radio = gr.Radio(
                choices=["Claude API", "로컬 (Ollama)"], value="Claude API", label="AI 제공자",
            )
            with gr.Group(visible=True) as claude_group:
                ai_model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Claude 모델",
                )
                ai_api_key_box = gr.Textbox(
                    label="Anthropic API 키 (선택)", type="password",
                    placeholder="sk-ant-... (비워두면 .env의 ANTHROPIC_API_KEY 사용)",
                )
            with gr.Group(visible=False) as ollama_group:
                gr.Markdown(
                    "[Ollama 설치](https://ollama.com/download) 후 터미널에서 "
                    f"`ollama pull {DEFAULT_OLLAMA_MODEL}` 실행 (한국어 지원, 약 4.8GB). "
                    "API 키 불필요, 인터넷 연결도 불필요합니다.",
                    elem_classes="note",
                )
                with gr.Row():
                    ai_ollama_model_box = gr.Textbox(
                        value=DEFAULT_OLLAMA_MODEL, label="Ollama 모델", scale=2,
                    )
                    ai_ollama_host_box = gr.Textbox(
                        value=DEFAULT_OLLAMA_HOST, label="Ollama 주소", scale=1,
                    )
            ai_insight_btn = gr.Button("AI 인사이트 생성", variant="primary")
            ai_insight_output = gr.Markdown(value="", elem_classes="note")
            ai_insight_state = gr.State(value=None)

            ai_provider_radio.change(
                fn=lambda p: (
                    gr.update(visible=(p == "Claude API")),
                    gr.update(visible=(p == "로컬 (Ollama)")),
                ),
                inputs=[ai_provider_radio],
                outputs=[claude_group, ollama_group],
            )

        with gr.Row():
            gr.HTML(
                '<p style="font-size:14px;font-weight:700;letter-spacing:0.09em;'
                'text-transform:uppercase;color:var(--body-text-color);'
                'padding:0 0 0 10px;border-left:3px solid var(--border-color-accent);'
                'margin:24px 0 8px 0;line-height:1.5;display:block;">내보내기</p>'
            )
            guide_btn_export = gr.Button("?", elem_classes="guide-btn", scale=0, min_width=36)
        with gr.Row():
            csv_download   = gr.File(label="예측 데이터  (CSV)")
            excel_download = gr.File(label="연구 보고서  (Excel)")
        with gr.Row():
            charts_download  = gr.File(label="분석 차트  (HTML / SVG)")
            session_download = gr.File(label="세션 파일  (JSON)")

        with gr.Row():
            gr.HTML(
                '<p style="font-size:14px;font-weight:700;letter-spacing:0.09em;'
                'text-transform:uppercase;color:var(--body-text-color);'
                'padding:0 0 0 10px;border-left:3px solid var(--border-color-accent);'
                'margin:24px 0 8px 0;line-height:1.5;display:block;">배치 처리 결과</p>'
            )
            guide_btn_batch = gr.Button("?", elem_classes="guide-btn", scale=0, min_width=36)
        batch_summary_df_out = gr.DataFrame(label="영상별 처리 요약")
        batch_zip_download   = gr.File(label="전체 결과 다운로드 (ZIP)")

        gr.HTML(
            '<p style="font-size:11px;color:var(--body-text-color-subdued);'
            'border-top:1px solid var(--border-color-primary);'
            'margin-top:14px;padding-top:10px;line-height:1.9;">'
            '실선 = 과거 이동 경로 &nbsp;|&nbsp; 점선 = 예측 궤적 &nbsp;|&nbsp;'
            ' 청색선 = 감지선 &nbsp;|&nbsp; 오렌지 폴리곤 = 분석 존 &nbsp;|&nbsp;'
            ' 각 색상은 고유 Track ID를 나타냅니다.'
            '</p>'
        )

    with gr.Column(elem_id="settings-modal", visible=False) as settings_modal:
        with gr.Column(elem_id="settings-panel"):
            _sec("기본 설정")
            gr.Markdown(
                "탐지·추적·예측의 핵심 파라미터입니다. YOLO 모델이 클수록 정확하지만 느려지고, "
                "시퀀스/예측 길이는 LSTM이 보는 과거 구간과 예측할 미래 구간의 길이(프레임)입니다.",
                elem_classes="note",
            )
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

            # ── Traffic Analysis ─────────────────────────────────────
            with gr.Accordion("교통 분석", open=False):
                gr.Markdown(
                    "가상 감지선 통과 계수·유량·속도 추정 등 도로 교통 분석 기능을 켜고 설정합니다. "
                    "(OD 행렬은 아래 '도시 공간 분석'에서 설정합니다.)",
                    elem_classes="note",
                )
                enable_traffic = gr.Checkbox(label="교통 분석 활성화", value=False)

                gr.Markdown(
                    "**감지선 정의** — 시작점(x1, y1)에서 끝점(x2, y2)으로 이어지는 가상 선을 정의합니다. "
                    "통과 차량을 자동으로 계수합니다.",
                    elem_classes="note",
                )

                # 4 counting line slots
                line_widgets = []
                _line_defaults = [
                    (True,  "Line 1", 100, 300, 800, 300),
                    (False, "Line 2",   0,   0,   0,   0),
                    (False, "Line 3",   0,   0,   0,   0),
                    (False, "Line 4",   0,   0,   0,   0),
                ]
                for _i, (_en, _lbl, _x1, _y1, _x2, _y2) in enumerate(_line_defaults):
                    with gr.Group():
                        with gr.Row():
                            _ln_en  = gr.Checkbox(label=f"Line {_i+1} 활성화", value=_en, scale=1)
                            _ln_lbl = gr.Textbox(value=_lbl, label="이름", scale=2)
                        with gr.Row():
                            _ln_x1 = gr.Number(value=_x1, label="x1 (시작)", step=1, scale=1)
                            _ln_y1 = gr.Number(value=_y1, label="y1 (시작)", step=1, scale=1)
                            _ln_x2 = gr.Number(value=_x2, label="x2 (끝)",   step=1, scale=1)
                            _ln_y2 = gr.Number(value=_y2, label="y2 (끝)",   step=1, scale=1)
                    line_widgets.append((_ln_en, _ln_lbl, _ln_x1, _ln_y1, _ln_x2, _ln_y2))

                gr.Markdown(
                    "**속도 추정 스케일** — 실거리와 픽셀 거리를 입력하면 km/h로 변환합니다.",
                    elem_classes="note",
                )
                with gr.Row():
                    ref_real_m = gr.Number(label="실거리 (m)", value=3.5, info="예: 차선폭 3.5 m")
                    ref_px     = gr.Number(label="픽셀 거리 (px)", value=100, info="해당 거리의 픽셀 수")
                enable_speed = gr.Checkbox(label="속도 추정 (km/h)", value=False)

            # ── Urban Analysis ───────────────────────────────────────
            with gr.Accordion("도시 공간 분석", open=False):
                gr.Markdown(
                    "밀도 열지도와 존별 체류시간·점유율 등 도시 공간 분석 기능을 켜고 설정합니다.",
                    elem_classes="note",
                )
                enable_urban   = gr.Checkbox(label="도시 분석 활성화", value=False)
                enable_heatmap = gr.Checkbox(label="밀도 열지도 생성", value=True)
                heatmap_classes = gr.CheckboxGroup(
                    choices=COCO_TRACKABLE,
                    value=["person"],
                    label="열지도 대상 클래스",
                )

                gr.Markdown(
                    "**존 정의** — 직사각형 분석 영역을 정의합니다. 좌상단(x1, y1)과 우하단(x2, y2) "
                    "픽셀 좌표를 입력하세요. 아래 OD 행렬과 공유됩니다.",
                    elem_classes="note",
                )

                # 4 zone slots
                zone_widgets = []
                for _i in range(4):
                    with gr.Group():
                        with gr.Row():
                            _z_en   = gr.Checkbox(label=f"Zone {chr(65+_i)} 활성화", value=False, scale=1)
                            _z_nm   = gr.Textbox(value=f"Zone {chr(65+_i)}", label="이름", scale=2)
                            _z_area = gr.Number(value=0.0, label="면적 (m²)", scale=1,
                                                info="밀도 계산용, 0이면 생략")
                        with gr.Row():
                            _z_x1 = gr.Number(value=0, label="x1 (좌상단)", step=1, scale=1)
                            _z_y1 = gr.Number(value=0, label="y1 (좌상단)", step=1, scale=1)
                            _z_x2 = gr.Number(value=0, label="x2 (우하단)", step=1, scale=1)
                            _z_y2 = gr.Number(value=0, label="y2 (우하단)", step=1, scale=1)
                    zone_widgets.append((_z_en, _z_nm, _z_x1, _z_y1, _z_x2, _z_y2, _z_area))

                gr.Markdown(
                    "**OD 행렬** — 위에서 활성화한 존들 사이의 이동(예: Zone A → Zone B)을 집계합니다. "
                    "**최소 2개 이상의 존을 활성화**해야 결과가 나오며, '도시 분석 활성화'와는 무관하게 "
                    "독립적으로 동작합니다.",
                    elem_classes="note",
                )
                enable_od = gr.Checkbox(label="OD 행렬 계산", value=False)

            # ── Calibration ──────────────────────────────────────────
            with gr.Accordion("캘리브레이션", open=False):
                gr.Markdown(
                    "픽셀→실세계 거리 변환 방식을 선택합니다. '기준거리' 선택 시 교통 분석에 입력한 실거리·픽셀거리로 m/px를 자동 계산합니다.",
                    elem_classes="note",
                )
                calib_mode = gr.Radio(
                    choices=["없음", "기준거리"],
                    value="없음",
                    label="캘리브레이션 모드",
                    info="'기준거리': 위의 실거리·픽셀거리로 m/px 자동 계산",
                )

            _sec("내보내기")
            gr.Markdown(
                "처리 결과를 저장할 형식을 선택합니다. CSV는 기본, Excel과 차트는 추가 분석용 보고서를 생성합니다.",
                elem_classes="note",
            )
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

            _sec("세션")
            gr.Markdown(
                "분석 설정을 JSON으로 저장·불러와 실험을 재현합니다. 연구자 메모도 함께 기록됩니다.",
                elem_classes="note",
            )
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

            settings_done_btn = gr.Button("완료", variant="primary")

    with gr.Column(elem_id="guide-modal", visible=False) as guide_modal:
        with gr.Column(elem_id="guide-panel"):
            _GUIDE_IMG_KW = dict(show_label=False, container=True, interactive=False,
                                 buttons=["fullscreen"])

            def _guide_img(filename):
                return gr.Image(value=str(GUIDE_IMG_DIR / filename), **_GUIDE_IMG_KW)

            with gr.Tabs() as guide_tabs:
                with gr.Tab("기본 설정", id=0):
                    gr.Markdown(
                        "탐지·추적·예측의 핵심 파라미터입니다. YOLO 모델이 클수록 정확하지만 느려지고, "
                        "시퀀스/예측 길이는 LSTM이 보는 과거 구간과 예측할 미래 구간의 길이(프레임)입니다. "
                        "이 설정은 '추적 결과'(어노테이션 영상·궤적 요약 이미지)에 직접 반영됩니다. "
                        "아래는 실제 '세부설정' 화면의 모습입니다.",
                        elem_classes="note",
                    )
                    _guide_img("00_basic_settings.png")
                    gr.Markdown("**직접 눌러보세요** (예시 — 실제 설정에는 적용되지 않습니다)",
                               elem_classes="note")
                    with gr.Group():
                        gr.Dropdown(
                            choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
                                     "yolo11l.pt", "yolo11x.pt"],
                            value="yolo11n.pt", label="YOLO 모델 (예시)", info="n = 빠름 / x = 정확",
                            interactive=True,
                        )
                        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="감지 신뢰도 임계값 (예시)",
                                  interactive=True)
                with gr.Tab("교통 분석", id=1):
                    gr.Markdown(
                        "가상 감지선 통과 계수·유량·속도 추정 등 도로 교통 분석 기능입니다. "
                        "**감지선 정의** — 시작점(x1, y1)에서 끝점(x2, y2)으로 이어지는 가상 선을 정의하면 "
                        "통과 차량을 자동으로 계수합니다. **속도 추정 스케일** — 실거리와 픽셀 거리를 입력하면 "
                        "km/h로 변환합니다. 결과 화면에서는 클래스별 통과 수·유량·속도 분포 차트와 함께 "
                        "Origin-Destination 행렬도 함께 표시됩니다(OD 행렬 자체는 '도시 공간 분석'에서 설정). "
                        "아래는 실제 '교통 분석' 설정 화면입니다 — Line 1처럼 활성화·이름·좌표를 입력합니다.",
                        elem_classes="note",
                    )
                    _guide_img("01_traffic.png")
                    gr.Markdown("**직접 눌러보세요** (예시 — 실제 설정에는 적용되지 않습니다)",
                               elem_classes="note")
                    with gr.Group():
                        with gr.Row():
                            gr.Checkbox(label="Line 1 활성화 (예시)", value=True, scale=1,
                                       interactive=True)
                            gr.Textbox(value="Line 1", label="이름 (예시)", scale=2,
                                      interactive=True)
                        with gr.Row():
                            gr.Number(value=100, label="x1 (예시)", scale=1, interactive=True)
                            gr.Number(value=300, label="y1 (예시)", scale=1, interactive=True)
                            gr.Number(value=800, label="x2 (예시)", scale=1, interactive=True)
                            gr.Number(value=300, label="y2 (예시)", scale=1, interactive=True)
                with gr.Tab("도시 공간 분석", id=2):
                    gr.Markdown(
                        "밀도 열지도와 존별 체류시간·점유율 등 도시 공간 분석 기능입니다. "
                        "**존 정의** — 직사각형 분석 영역을 정의합니다. 좌상단(x1, y1)과 우하단(x2, y2) "
                        "픽셀 좌표를 입력하세요. **OD 행렬** — 활성화한 존들 사이의 이동(예: Zone A → Zone B)을 "
                        "집계합니다. 최소 2개 이상의 존을 활성화해야 결과가 나오며, '도시 분석 활성화'와는 "
                        "무관하게 독립적으로 동작합니다. 아래는 실제 '도시 공간 분석' 설정 화면입니다.",
                        elem_classes="note",
                    )
                    _guide_img("02_urban.png")
                    gr.Markdown("**직접 눌러보세요** (예시 — 실제 설정에는 적용되지 않습니다)",
                               elem_classes="note")
                    with gr.Group():
                        gr.Checkbox(label="밀도 열지도 생성 (예시)", value=True, interactive=True)
                        gr.CheckboxGroup(
                            choices=["person", "car", "bicycle", "motorcycle"],
                            value=["person"], label="열지도 대상 클래스 (예시)",
                            interactive=True,
                        )
                with gr.Tab("캘리브레이션", id=3):
                    gr.Markdown(
                        "픽셀→실세계 거리 변환 방식을 선택합니다. '기준거리' 선택 시 교통 분석에 입력한 "
                        "실거리·픽셀거리로 m/px를 자동 계산합니다. 아래는 실제 '캘리브레이션' 설정 화면입니다.",
                        elem_classes="note",
                    )
                    _guide_img("03_calibration.png")
                    gr.Markdown("**직접 눌러보세요** (예시 — 실제 설정에는 적용되지 않습니다)",
                               elem_classes="note")
                    gr.Radio(choices=["없음", "기준거리"], value="없음",
                            label="캘리브레이션 모드 (예시)", interactive=True)
                with gr.Tab("내보내기", id=4):
                    gr.Markdown(
                        "처리 결과를 저장할 형식을 선택합니다. CSV는 기본, Excel과 차트는 추가 분석용 "
                        "보고서를 생성합니다. 결과 화면 하단의 '내보내기' 섹션에서 각 파일을 내려받을 수 "
                        "있습니다. 아래는 실제 '내보내기' 설정 화면입니다.",
                        elem_classes="note",
                    )
                    _guide_img("04_export.png")
                    gr.Markdown("**직접 눌러보세요** (예시 — 실제 설정에는 적용되지 않습니다)",
                               elem_classes="note")
                    gr.Radio(choices=["CSV만", "CSV + Excel", "CSV + Excel + 차트"],
                            value="CSV만", label="내보내기 형식 (예시)", interactive=True)
                with gr.Tab("세션", id=5):
                    gr.Markdown(
                        "분석 설정을 JSON으로 저장·불러와 실험을 재현합니다. 연구자 메모도 함께 기록됩니다. "
                        "아래는 실제 '세션' 설정 화면입니다 — 메모를 적어두면 세션 파일과 함께 저장됩니다.",
                        elem_classes="note",
                    )
                    _guide_img("05_session.png")
                with gr.Tab("트랙 요약", id=6):
                    gr.Markdown(
                        "추적된 개별 객체(트랙)별 이동 경로 요약입니다. 진입·퇴장 프레임, 이동 거리, "
                        "속도를 확인할 수 있습니다. 아래는 결과 화면의 '트랙 요약' 표 예시입니다(영상에서 "
                        "추적된 트랙이 없으면 빈 표로 표시됩니다).",
                        elem_classes="note",
                    )
                    _guide_img("06_track_summary.png")
                with gr.Tab("AI 인사이트", id=7):
                    gr.Markdown(
                        "Claude API 또는 로컬 AI(Ollama)로 분석 결과를 한국어로 해설합니다. "
                        "주목할 데이터는 강조점으로 모아 보여주고(Claude만), 해당 차트·표에도 함께 "
                        "표시됩니다. 아래는 실제 'AI 인사이트' 결과 화면입니다.",
                        elem_classes="note",
                    )
                    _guide_img("07_ai_insight.png")
                    gr.Markdown("**직접 눌러보세요** (예시 — 실제로 전환되지 않습니다)",
                               elem_classes="note")
                    gr.Radio(choices=["Claude API", "로컬 (Ollama)"], value="Claude API",
                            label="AI 제공자 (예시)", interactive=True)
                with gr.Tab("배치 처리", id=8):
                    gr.Markdown(
                        "여러 영상을 한 번에 처리할 수 있습니다(홈 화면의 '배치 처리' 탭). 배치로 처리한 "
                        "경우, 결과 화면 상단에서 영상을 골라 이 페이지 전체(차트·표·AI 인사이트)를 그 "
                        "영상 기준으로 바꿔 볼 수 있습니다. 기본값은 마지막으로 처리된 영상입니다. 영상별 "
                        "처리 결과 요약과 전체 결과 ZIP 다운로드는 결과 화면의 '배치 처리 결과' 섹션에서 "
                        "확인할 수 있습니다. 아래는 홈 화면의 '배치 처리' 업로드 탭입니다.",
                        elem_classes="note",
                    )
                    _guide_img("08_batch_upload.png")
            guide_close_btn = gr.Button("닫기")

    # ── Flatten widget lists ──────────────────────────────────────────────
    _line_inputs_flat = [w for tup in line_widgets for w in tup]   # 4×6 = 24
    _zone_inputs_flat = [w for tup in zone_widgets for w in tup]   # 4×7 = 28

    _all_outputs = [
        video_output, csv_download, summary_img, status_box,
        traffic_count_plot, traffic_speed_plot,
        od_df_out,
        heatmap_img_out, zone_df_out,
        track_summary_df_out,
        excel_download, charts_download, session_download,
        batch_summary_df_out, batch_zip_download,
        ai_insight_state,
        batch_packages_state, batch_video_select,
    ]  # 18 outputs

    def _maybe_switch_to_results(video_path):
        if video_path:
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False)

    run_btn.click(
        fn=process_videos,
        inputs=[
            video_input, batch_files,
            yolo_model, conf_thresh, seq_len, pred_len,
            lstm_mode, class_filter,
            session_notes,
            enable_traffic,
            *_line_inputs_flat,
            ref_real_m, ref_px, enable_speed, enable_od,
            enable_urban, enable_heatmap, heatmap_classes,
            *_zone_inputs_flat,
            calib_mode,
            export_format, enable_mc_dropout, chart_format,
        ],
        outputs=_all_outputs,
    ).then(
        fn=_maybe_switch_to_results,
        inputs=[video_output],
        outputs=[home_screen, results_screen],
    )

    session_load.upload(
        fn=restore_session,
        inputs=[session_load],
        outputs=[
            yolo_model, conf_thresh, seq_len, pred_len,
            lstm_mode, class_filter, session_notes,
            enable_traffic,
            *_line_inputs_flat,
            enable_speed, enable_od,
            enable_urban, enable_heatmap, heatmap_classes,
            *_zone_inputs_flat,
            export_format, enable_mc_dropout, chart_format,
        ],
    )

    video_input.clear(
        fn=lambda: (None,) * 17 + (gr.update(choices=[], value=None),),
        inputs=[],
        outputs=_all_outputs,
    )

    ai_insight_btn.click(
        fn=lambda: gr.update(interactive=False, value="AI 인사이트 생성 중..."),
        inputs=[],
        outputs=[ai_insight_btn],
    ).then(
        fn=run_ai_insight,
        inputs=[
            ai_insight_state, ai_provider_radio,
            ai_model_dropdown, ai_api_key_box,
            ai_ollama_model_box, ai_ollama_host_box,
        ],
        outputs=[
            ai_insight_output, traffic_count_plot, traffic_speed_plot,
            zone_df_out, track_summary_df_out,
        ],
    ).then(
        fn=lambda: gr.update(interactive=True, value="AI 인사이트 생성"),
        inputs=[],
        outputs=[ai_insight_btn],
    )

    batch_view_btn.click(
        fn=view_batch_video,
        inputs=[batch_packages_state, batch_video_select],
        outputs=[
            video_output, csv_download, summary_img, status_box,
            traffic_count_plot, traffic_speed_plot, od_df_out,
            heatmap_img_out, zone_df_out, track_summary_df_out,
            excel_download, charts_download,
            ai_insight_state, ai_insight_output,
        ],
    )

    monitor_timer.tick(
        fn=_update_monitor,
        inputs=[],
        outputs=[monitor_summary, cpu_plot, ram_plot, gpu_plot],
    )

    # ── Settings modal open/close ───────────────────────────────────────
    open_settings_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[], outputs=[settings_modal],
    )
    settings_done_btn.click(
        fn=lambda: gr.update(visible=False),
        inputs=[], outputs=[settings_modal],
    )

    # ── Guide modal open/close + per-section deep links ─────────────────
    open_guide_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[], outputs=[guide_modal],
    )
    guide_close_btn.click(
        fn=lambda: gr.update(visible=False),
        inputs=[], outputs=[guide_modal],
    )

    def _open_guide_at(n):
        return lambda: (gr.update(visible=True), gr.update(selected=n))

    guide_btn_tracking.click(fn=_open_guide_at(0), inputs=[], outputs=[guide_modal, guide_tabs])
    guide_btn_traffic.click(fn=_open_guide_at(1), inputs=[], outputs=[guide_modal, guide_tabs])
    guide_btn_urban.click(fn=_open_guide_at(2), inputs=[], outputs=[guide_modal, guide_tabs])
    guide_btn_track_summary.click(fn=_open_guide_at(6), inputs=[], outputs=[guide_modal, guide_tabs])
    guide_btn_ai.click(fn=_open_guide_at(7), inputs=[], outputs=[guide_modal, guide_tabs])
    guide_btn_export.click(fn=_open_guide_at(4), inputs=[], outputs=[guide_modal, guide_tabs])
    guide_btn_batch.click(fn=_open_guide_at(8), inputs=[], outputs=[guide_modal, guide_tabs])

    # ── New analysis: reset uploaded video/results, keep settings, go home ──
    _RESET_ALL_OUTPUTS = (None,) * 17 + (gr.update(choices=[], value=None),)

    new_analysis_btn.click(
        fn=lambda: (None, None) + _RESET_ALL_OUTPUTS + (gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[video_input, batch_files] + _all_outputs + [home_screen, results_screen],
    )

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware

    class _SABHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            return response

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(_SABHeadersMiddleware)
    gr.mount_gradio_app(fastapi_app, demo, path="/",
                        theme=gr.themes.Monochrome(), css=css)

    print("[서버] http://127.0.0.1:7860")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=7860)
