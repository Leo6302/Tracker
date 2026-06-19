# Object Tracking + Trajectory Prediction

YOLOv11 기반 다중 객체 추적과 LSTM 궤적 예측을 결합한 교통·도시 연구용 분석 툴입니다.  
Gradio 웹 UI로 동작하며, 단일 영상부터 배치 처리까지 지원합니다.

---

## 주요 기능

### 추적 · 예측
- **YOLOv11** (n / s / m / l / x) 다중 객체 감지 — 15개 COCO 클래스 선택 추적
- **ByteTrack** 기반 안정적 다중 객체 추적
- **LSTM 궤적 예측** — 과거 시퀀스로 미래 경로 예측 (사전학습 / 온라인 파인튜닝)
- **MC-Dropout 불확실성 추정** — 예측 신뢰 구간 출력

### 교통 분석
- 가상 감지선 통과 계수 (차종별 유량 · 차두시간)
- 픽셀→km/h 속도 추정 (EMA 스무딩, P85 출력)
- Origin-Destination 행렬
- Greenshields 기반 혼잡 감지

### 도시 공간 분석
- 적응형 KDE 밀도 열지도
- 존 체류시간 · 점유율 · 피크 윈도우 분석

### 배치 처리
- 여러 영상을 한 번에 업로드하여 순차 처리
- 전체 진행 현황 + 영상별 프레임 진행률 동시 표시
- 처리 완료 후 ZIP 일괄 다운로드

### 내보내기
- 예측 데이터 CSV
- 다중 시트 Excel 연구 보고서
- Plotly 인터랙티브 HTML 차트
- JSON 세션 저장 / 불러오기 (분석 설정 재현)

---

## 요구 사항

| 항목 | 최소 사양 |
|------|-----------|
| Python | 3.10 이상 |
| RAM | 8 GB |
| GPU | NVIDIA CUDA (선택 — 없으면 CPU로 동작) |
| OS | Windows / macOS / Linux |

---

## 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/Leo6302/Tracker.git
cd Tracker

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 실행
python app.py
```

브라우저에서 `http://127.0.0.1:7860` 으로 접속합니다.

> **모델 파일** — `yolo11n.pt`(루트)와 `models/lstm_pretrained.pt`가 저장소에 포함되어 있어 별도 다운로드 없이 바로 실행됩니다.  
> 사전학습 모델이 없을 경우 첫 실행 시 자동으로 생성됩니다 (1~2분 소요).

---

## 프로젝트 구조

```
Tracker/
├── app.py                    # Gradio 앱 진입점
├── config.yaml               # 모델·학습 기본 설정
├── yolo11n.pt                # YOLOv11 nano 가중치
├── models/
│   └── lstm_pretrained.pt    # LSTM 사전학습 가중치
├── requirements.txt
└── src/
    ├── pipeline.py           # 전체 처리 파이프라인
    ├── device_utils.py       # CUDA / MPS / CPU 자동 감지
    ├── tracking/
    │   └── tracker.py        # ByteTrack 래퍼
    ├── prediction/
    │   ├── model.py          # TrajectoryLSTM
    │   ├── trainer.py        # 사전학습 / 파인튜닝
    │   └── preprocessor.py   # 시퀀스 빌드 · 역정규화
    ├── analysis/
    │   ├── counting_line.py  # 가상 감지선 계수
    │   ├── speed_estimator.py
    │   ├── od_matrix.py
    │   ├── density_heatmap.py
    │   ├── zone_analyzer.py
    │   ├── congestion.py
    │   ├── calibration.py
    │   └── session.py        # 세션 저장 / 불러오기
    └── visualization/
        ├── renderer.py       # 프레임 오버레이
        ├── exporter.py       # 영상 · CSV · 궤적 이미지
        └── stats_exporter.py # Excel · Plotly 차트
```

---

## 설정 (`config.yaml`)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `seq_len` | 20 | LSTM 입력 과거 프레임 수 |
| `pred_len` | 10 | 예측 미래 프레임 수 |
| `conf_thresh` | 0.5 | YOLO 감지 신뢰도 임계값 |
| `hidden_size` | 256 | LSTM 은닉층 크기 |
| `num_layers` | 3 | LSTM 레이어 수 |
| `dropout` | 0.2 | 드롭아웃 비율 |
| `finetune_fraction` | 0.3 | 파인튜닝에 사용할 영상 앞부분 비율 |

---

## 시스템 모니터링

처리 중 실시간으로 CPU · RAM · GPU 사용률을 확인할 수 있습니다.  
우측 패널의 **상세 모니터링** 아코디언을 열면 최근 60초 추이 그래프가 표시됩니다.

---

## 수정·개선 내역 (재현 가능한 상세 기록)

> 이 섹션은 의도적으로 자세하게 적혀 있습니다. 같은 프로젝트의 이전 버전(아래 변경이 아직 적용되지 않은 코드)을 가진 사람이 이 README를 Claude(또는 다른 코딩 에이전트)에게 주고
> "이 문서대로 고쳐줘"라고 하면, 증상 → 원인 → 검증 방법 → 실제 코드까지 보고 동일한 수정을 재현할 수 있도록 정리했습니다.
> 이미 이 코드를 그대로 받았다면 아래 내용은 모두 적용되어 있으니 참고용으로만 보면 됩니다. 파일 경로는 모두 `Tracker/` 기준입니다.

---

### 9. 감지선·존 정의가 JSON 직접 입력 방식이라 조작이 불편함

- **증상**: "교통 분석" 아코디언에 감지선을, "도시 공간 분석" 아코디언에 존을 설정하려면 `[{"label": "Line 1", "x1": 100, "y1": 300, "x2": 800, "y2": 300}]` 같은 JSON을 `gr.Textbox`에 직접 손으로 입력해야 했음. 좌표 하나 바꾸는 것도 JSON 배열 구조 안을 눈으로 찾아 편집해야 하고, 중괄호 하나가 빠지면 파싱 실패 시 에러 메시지 없이 빈 설정(감지선 없음)으로 조용히 실행돼 결과가 이상해짐. 존의 경우 polygon 좌표 배열(`[[100,100],[300,100],[300,300],[100,300]]`)을 써야 해 특히 불편함.
- **원인**: `app.py`의 `counting_lines_json = gr.Textbox(...)`, `zones_json = gr.Textbox(...)`, `zone_areas_json = gr.Textbox(...)`가 JSON 문자열을 그대로 받아 `process_videos()` 안에서 `json.loads()`로 파싱하는 방식. UI에서 유효성 검사도 없었고, 여러 항목(복수 감지선)을 추가·삭제하는 인터랙션도 없었음.
- **검증 방법**: 감지선 textbox에 `{"label"` (닫는 대괄호 누락) 를 넣고 처리 → 에러 없이 `counting_lines=[]`로 실행되고 결과에서 감지선 집계가 통째로 사라지는 것을 확인.
- **해결**: 감지선 4슬롯·존 4슬롯을 각각 체크박스(활성화) + 이름 + x1/y1/x2/y2 숫자 입력으로 교체. 존은 직사각형으로 단순화(좌상단 x1,y1 — 우하단 x2,y2)하고, 내부에서 자동으로 4점 polygon으로 변환. 소스 파일은 `app.py` 한 곳만 바꾸면 됨(파이프라인에 전달하는 JSON 구조는 그대로 유지됨).

**핵심 구조: 위젯을 리스트로 관리하고 나중에 flat하게 풀기**

Gradio의 `run_btn.click(inputs=[...])` 는 평범한 Python 리스트이므로 `*` 언패킹이 그대로 동작함. 이 패턴 덕분에 루프로 위젯을 만들어 리스트에 담은 뒤, 이벤트 핸들러에 한 번에 풀어 넣을 수 있음.

```python
# app.py — gr.Blocks 안, 교통 분석 아코디언
line_widgets = []   # 리스트에 (enable, label, x1, y1, x2, y2) 튜플로 쌓기
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
```

```python
# app.py — 도시 공간 분석 아코디언 (존은 튜플이 7개: area 포함)
zone_widgets = []
for _i in range(4):
    with gr.Group():
        with gr.Row():
            _z_en   = gr.Checkbox(label=f"Zone {chr(65+_i)} 활성화", value=False, scale=1)
            _z_nm   = gr.Textbox(value=f"Zone {chr(65+_i)}", label="이름", scale=2)
            _z_area = gr.Number(value=0.0, label="면적 (m²)", scale=1, info="밀도 계산용, 0이면 생략")
        with gr.Row():
            _z_x1 = gr.Number(value=0, label="x1 (좌상단)", step=1, scale=1)
            _z_y1 = gr.Number(value=0, label="y1 (좌상단)", step=1, scale=1)
            _z_x2 = gr.Number(value=0, label="x2 (우하단)", step=1, scale=1)
            _z_y2 = gr.Number(value=0, label="y2 (우하단)", step=1, scale=1)
    zone_widgets.append((_z_en, _z_nm, _z_x1, _z_y1, _z_x2, _z_y2, _z_area))
```

```python
# app.py — gr.Blocks 안, 이벤트 핸들러 직전에서 flat하게 풀기
_line_inputs_flat = [w for tup in line_widgets for w in tup]  # 4×6 = 24
_zone_inputs_flat = [w for tup in zone_widgets for w in tup]  # 4×7 = 28

run_btn.click(
    fn=process_videos,
    inputs=[
        video_input, batch_files,
        yolo_model, conf_thresh, seq_len, pred_len, lstm_mode, class_filter,
        session_notes,
        enable_traffic,
        *_line_inputs_flat,          # 24개 언패킹
        ref_real_m, ref_px, enable_speed, enable_od,
        enable_urban, enable_heatmap, heatmap_classes,
        *_zone_inputs_flat,          # 28개 언패킹
        calib_mode,
        export_format, enable_mc_dropout, chart_format,
    ],
    outputs=_all_outputs,
)
```

**`process_videos` 함수 시그니처 변경** — 기존 `counting_lines_json, zones_json, zone_areas_json` 3개 파라미터 대신 개별 위젯 73개로 교체:

```python
# app.py
def process_videos(
    video_input, batch_files,
    yolo_model, conf_thresh, seq_len, pred_len, lstm_mode, class_filter,
    session_notes,
    enable_traffic,
    l1_en, l1_lbl, l1_x1, l1_y1, l1_x2, l1_y2,   # Line 1
    l2_en, l2_lbl, l2_x1, l2_y1, l2_x2, l2_y2,   # Line 2
    l3_en, l3_lbl, l3_x1, l3_y1, l3_x2, l3_y2,   # Line 3
    l4_en, l4_lbl, l4_x1, l4_y1, l4_x2, l4_y2,   # Line 4
    ref_real_m, ref_px, enable_speed, enable_od,
    enable_urban, enable_heatmap, heatmap_classes,
    z1_en, z1_nm, z1_x1, z1_y1, z1_x2, z1_y2, z1_area,   # Zone A
    z2_en, z2_nm, z2_x1, z2_y1, z2_x2, z2_y2, z2_area,   # Zone B
    z3_en, z3_nm, z3_x1, z3_y1, z3_x2, z3_y2, z3_area,   # Zone C
    z4_en, z4_nm, z4_x1, z4_y1, z4_x2, z4_y2, z4_area,   # Zone D
    calib_mode,
    export_format, enable_mc_dropout, chart_format,
    progress=gr.Progress(track_tqdm=True),
):
    # 함수 첫머리에서 JSON 빌드
    _line_raw = [
        (l1_en, l1_lbl, l1_x1, l1_y1, l1_x2, l1_y2),
        (l2_en, l2_lbl, l2_x1, l2_y1, l2_x2, l2_y2),
        (l3_en, l3_lbl, l3_x1, l3_y1, l3_x2, l3_y2),
        (l4_en, l4_lbl, l4_x1, l4_y1, l4_x2, l4_y2),
    ]
    counting_lines = [
        {"label": (lbl or f"Line {i+1}"),
         "x1": int(x1 or 0), "y1": int(y1 or 0),
         "x2": int(x2 or 0), "y2": int(y2 or 0)}
        for i, (en, lbl, x1, y1, x2, y2) in enumerate(_line_raw) if en
    ] if enable_traffic else []

    _zone_raw = [
        (z1_en, z1_nm, z1_x1, z1_y1, z1_x2, z1_y2, z1_area),
        # ... z2~z4 동일
    ]
    zones = []
    zone_areas = {}
    for i, (en, nm, x1, y1, x2, y2, area) in enumerate(_zone_raw):
        if not en:
            continue
        nm = nm or f"Zone {chr(65 + i)}"
        x1, y1, x2, y2 = int(x1 or 0), int(y1 or 0), int(x2 or 0), int(y2 or 0)
        zones.append({"name": nm, "polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]})
        if area:
            zone_areas[nm] = float(area)
    # 이후 counting_lines, zones, zone_areas를 analysis_config에 그대로 전달
```

**`restore_session` 반환값 변경** — 기존 19개 → 68개. 저장된 JSON을 개별 위젯 값으로 분해해 돌려줌:

```python
# app.py
def restore_session(session_file):
    if not session_file:
        return (gr.skip(),) * 68   # 19 → 68로 변경
    ...
    # counting_lines JSON → 4슬롯으로 분해
    lines = cfg.counting_lines or []
    line_vals = [(False, f"Line {i+1}", 0, 0, 0, 0) for i in range(4)]
    for i, ln in enumerate(lines[:4]):
        line_vals[i] = (True, ln.get('label', f'Line {i+1}'),
                        int(ln.get('x1', 0)), int(ln.get('y1', 0)),
                        int(ln.get('x2', 0)), int(ln.get('y2', 0)))
    line_flat = [v for t in line_vals for v in t]   # 24개

    # zones JSON → 4슬롯으로 분해 (polygon bbox → x1,y1,x2,y2)
    zones_list = cfg.zones or []
    zone_areas_dict = cfg.zone_areas or {}
    zone_vals = [(False, f"Zone {chr(65+i)}", 0, 0, 0, 0, 0.0) for i in range(4)]
    for i, zone in enumerate(zones_list[:4]):
        nm = zone.get('name', f'Zone {chr(65+i)}')
        poly = zone.get('polygon', [[0,0],[0,0],[0,0],[0,0]])
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        zone_vals[i] = (True, nm, min(xs), min(ys), max(xs), max(ys),
                        float(zone_areas_dict.get(nm, 0.0)))
    zone_flat = [v for t in zone_vals for v in t]   # 28개

    return (
        cfg.yolo_model, cfg.conf_thresh, cfg.seq_len, cfg.pred_len,
        lstm_mode_label, cfg.class_filter, cfg.notes or '',
        cfg.enable_traffic,
        *line_flat,          # 24
        cfg.enable_speed, cfg.enable_od,
        cfg.enable_urban, cfg.enable_heatmap, cfg.heatmap_classes,
        *zone_flat,          # 28
        cfg.export_format, cfg.enable_mc_dropout, cfg.chart_format,
    )  # 합계 68개

# session_load.upload의 outputs도 동일하게 68개로 맞춰야 함
session_load.upload(
    fn=restore_session,
    inputs=[session_load],
    outputs=[
        yolo_model, conf_thresh, seq_len, pred_len,
        lstm_mode, class_filter, session_notes,
        enable_traffic,
        *_line_inputs_flat,          # 24
        enable_speed, enable_od,
        enable_urban, enable_heatmap, heatmap_classes,
        *_zone_inputs_flat,          # 28
        export_format, enable_mc_dropout, chart_format,
    ],
)
```

- **주의 (재현 시 흔히 빠지는 함정)**:
  - `zone_widgets` 튜플 순서가 `(_z_en, _z_nm, _z_x1, _z_y1, _z_x2, _z_y2, _z_area)`라면, `process_videos` 파라미터에서 `z1_en, z1_nm, z1_x1, z1_y1, z1_x2, z1_y2, z1_area` 순서와 **정확히 일치**해야 함. 순서가 하나라도 틀리면 이름란에 숫자가, 좌표란에 문자가 들어가는 조용한 오동작이 발생함.
  - `gr.skip()` 개수는 `session_load.upload(outputs=[...])` 의 출력 위젯 개수와 **정확히 같아야** 함 (이번엔 68). 68이 아니면 Gradio가 런타임 에러를 냄.
  - 기존에 `counting_lines_json` / `zones_json` / `zone_areas_json` 텍스트박스를 `run_btn.click(inputs=[...])` 에 넣고 있었다면, 이 3개를 제거하고 24+28개로 교체해야 함. 하나라도 빠지거나 중복되면 `process_videos` 파라미터 수와 어긋나 에러가 남.
- **파일**: `app.py`

---

### 10. 교통 분석 결과가 숫자 테이블만 있어 한눈에 파악이 어려움

- **증상**: 처리 후 "교통 분석 결과" 아코디언을 열면 숫자가 채워진 DataFrame 테이블(감지선·클래스·통과수·유량·차두시간·속도 등)만 표시됨. 여러 클래스 간 통과 수 크기 차이, 속도 분포의 이상치 유무 같은 정보를 눈으로 파악하려면 숫자를 하나씩 읽어야 했음. 내보내기에서 "CSV + Excel + 차트"를 선택하면 별도 HTML 파일로 Plotly 차트가 생성되긴 했지만 파일을 따로 열어야 했음.
- **원인**: `process_videos()`가 `traffic_df`, `speed_df`를 `pd.DataFrame`으로 만들어 `gr.DataFrame` 컴포넌트에 반환. 결과 화면에서 차트를 렌더링하는 코드가 없었음.
- **해결**: 두 개의 matplotlib 차트 생성 함수를 추가하고 `gr.DataFrame` 출력을 `gr.Plot` 출력 두 개로 교체.

**추가 함수 1 — 통과 수 · 유량 막대 차트**:

```python
# app.py — _update_monitor() 이후에 추가

def _make_traffic_count_chart(counting_data):
    """감지선별 클래스 통과 수·유량 그룹 막대 차트를 matplotlib Figure로 반환."""
    if not counting_data:
        return None
    all_cls = sorted({cls for line in counting_data for cls in line.get('counts', {})})
    if not all_cls:
        return None

    n_lines = len(counting_data)
    n_cls = len(all_cls)
    x = np.arange(n_cls)
    width = min(0.6 / max(n_lines, 1), 0.35)
    palette = [plt.cm.tab10(i / 10) for i in range(n_lines)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, n_cls * 2 + 3), 4))
    for i, line in enumerate(counting_data):
        lbl = line.get('label', f'Line {i+1}')
        offset = (i - (n_lines - 1) / 2) * width
        counts = [line.get('counts', {}).get(cls, 0) for cls in all_cls]
        flows  = [line.get('flow_rates_veh_hr', {}).get(cls, 0) for cls in all_cls]
        ax1.bar(x + offset, counts, width, label=lbl, color=palette[i], alpha=0.85)
        ax2.bar(x + offset, flows,  width, label=lbl, color=palette[i], alpha=0.85)

    for ax, title, ylabel in [
        (ax1, '클래스별 통과 수', '통과 수'),
        (ax2, '클래스별 유량 (대/시)', '유량 (대/시)'),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(all_cls, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
```

**추가 함수 2 — 속도 분포 박스플롯**:

```python
def _make_speed_chart(speed_data):
    """클래스별 속도 분포 박스플롯을 matplotlib Figure로 반환 (원시 트랙 샘플 사용)."""
    if not speed_data:
        return None
    track_speeds = speed_data.get('track_speeds', {})
    track_cls    = speed_data.get('track_cls', {})
    if not track_speeds:
        return None

    by_class: dict = {}
    for tid, speeds in track_speeds.items():
        cls = track_cls.get(tid) or track_cls.get(str(tid), 'unknown')
        by_class.setdefault(cls, []).extend(speeds)

    labels = [cls for cls in sorted(by_class) if len(by_class[cls]) >= 2]
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.8 + 2), 4))
    palette = [plt.cm.tab10(i / 10) for i in range(len(labels))]
    bp = ax.boxplot(
        [by_class[cls] for cls in labels],
        labels=labels, patch_artist=True,
        medianprops={'color': '#c0392b', 'linewidth': 2},
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5},
    )
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_ylabel('속도 (km/h)', fontsize=9)
    ax.set_title('클래스별 속도 분포', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
```

**`process_videos` 반환값 · 출력 위젯 변경** — 기존 `traffic_df`(DataFrame 1개) → `traffic_count_fig`, `traffic_speed_fig`(Plot 2개):

```python
# app.py — process_videos() 반환 직전
counting_data = last_analysis.get('counting_lines', [])
speed_data    = last_analysis.get('speed', {})
traffic_count_fig = _make_traffic_count_chart(counting_data)
traffic_speed_fig = _make_speed_chart(speed_data)

return (
    last_result['video'],
    last_result['csv'],
    last_result['trajectory_img'],
    status,
    traffic_count_fig,   # ← 기존 traffic_df 대체
    traffic_speed_fig,   # ← 신규 추가
    od_df,
    last_result.get('heatmap_img'),
    zone_df,
    pd.DataFrame(last_result.get('track_summary', [])),
    last_result.get('excel'),
    last_result.get('charts'),
    session_out,
    batch_summary_df,
    batch_zip_path,
)  # 14개 → 15개
```

```python
# app.py — 우측 패널의 교통 분석 결과 섹션
with gr.Accordion("교통 분석 결과", open=False):
    gr.Markdown("감지선 통과 통계 — 클래스별 계수·유량·속도 분포를 차트로 표시합니다.",
                elem_classes="note")
    traffic_count_plot = gr.Plot(label="통과 수 / 유량")   # ← gr.DataFrame 대체
    traffic_speed_plot = gr.Plot(label="속도 분포")        # ← 신규
    gr.Markdown("Origin-Destination 행렬", elem_classes="note")
    od_df_out = gr.DataFrame(label="OD 행렬")             # OD는 행렬 구조라 표로 유지

# _all_outputs 리스트도 동일하게 반영
_all_outputs = [
    video_output, csv_download, summary_img, status_box,
    traffic_count_plot, traffic_speed_plot,    # ← 변경
    od_df_out,
    heatmap_img_out, zone_df_out,
    track_summary_df_out,
    excel_download, charts_download, session_download,
    batch_summary_df_out, batch_zip_download,
]  # 14 → 15개
```

- **주의 (재현 시 흔히 빠지는 함정)**:
  - `_make_traffic_count_chart`와 `_make_speed_chart`는 `matplotlib.use('Agg')` 이후에 정의해야 함 (이미 파일 상단에 `matplotlib.use('Agg')`가 있으면 OK). 백엔드를 Agg로 고정하지 않으면 서버 환경에서 `cannot connect to X server` 에러가 날 수 있음.
  - 반환 타입이 `None`인 경우(`counting_data`가 비어있는 등)에도 `gr.Plot`은 조용히 빈 상태로 표시됨 — 별도 처리 불필요.
  - 출력 개수가 14 → 15개로 늘었으므로, 조기 반환 경로(`"영상을 먼저 업로드해주세요"`)와 `video_input.clear()`의 자리 채우기도 같이 늘려야 함:
    - 조기 반환: `(None, None, None, "영상을 먼저 업로드해주세요.") + (None,) * 11`  (10 → 11)
    - clear: `fn=lambda: (None,) * 15`  (14 → 15)
  - `import numpy as np`를 `app.py` 최상단에 추가해야 함 (`np.arange`, `plt.cm.tab10` 등이 차트 함수에서 사용됨).
- **파일**: `app.py`

#### 후속 수정 — 한글 깨짐·고리 아티팩트·호버 툴팁 문제로 matplotlib → Plotly 전환

> 항목 10에서 구현한 matplotlib 차트에서 세 가지 추가 문제가 발견되어 두 함수를 Plotly로 교체.

- **증상 1 — 한글 깨짐**: "클래스별 통과 수", "통과 수", "유량 (대/시)", "속도 (km/h)", "클래스별 속도 분포" 등 차트 제목·축 레이블에 한글이 모두 □□□ 박스 문자로 표시됨.
- **증상 2 — 고리 아티팩트**: 속도 분포 박스플롯에서 일부 클래스(예: bus)의 박스 상단에 여러 개의 이상치(outlier) 마커가 겹쳐 쌓여 고리·도넛 모양의 시각적 잡음이 발생함.
- **증상 3 — 제한적 정보**: 차트 위에 커서를 올려도 정확한 수치(통과 수, 유량 값, 속도 값)가 표시되지 않고 matplotlib 기본 좌표값만 나오거나 아무것도 표시되지 않음.
- **원인**: 세 증상 모두 matplotlib 백엔드에서 정적 PNG 이미지로 렌더링하기 때문에 발생.
  1. matplotlib은 시스템에서 한글 폰트를 자동으로 찾지 못해 박스 문자로 대체.
  2. matplotlib `ax.boxplot`의 `flierprops` 마커들이 같은 y값 근처에 수천 개 겹치면 겹침 효과로 고리처럼 보임.
  3. PNG 이미지는 호버 인터랙션이 없음 — `gr.Plot`이 matplotlib Figure를 PNG로 인코딩해 표시하므로 어떤 인터랙션도 불가.
- **해결**: `_make_traffic_count_chart`와 `_make_speed_chart` 두 함수를 matplotlib → **Plotly**로 교체. `gr.Plot` 컴포넌트는 matplotlib Figure와 Plotly Figure를 모두 지원하므로 출력 위젯 타입은 변경 불필요. Plotly는 브라우저에서 렌더링하므로 한글 자동 지원, 이상치 처리 방식이 다름(개별 점), 호버 툴팁 네이티브 지원.

**임포트 추가** — `app.py` 상단의 matplotlib 임포트 바로 아래에:

```python
# app.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ↓ 아래 3줄 추가
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
```

**교체 함수 1 — 통과 수 · 유량 막대 차트 (Plotly 버전)**:

```python
# app.py — _make_traffic_count_chart 전체를 아래로 교체
def _make_traffic_count_chart(counting_data):
    """감지선별 클래스 통과 수·유량 그룹 막대 차트 (Plotly, 인터랙티브)."""
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
```

**교체 함수 2 — 속도 분포 박스플롯 (Plotly 버전)**:

```python
# app.py — _make_speed_chart 전체를 아래로 교체
def _make_speed_chart(speed_data):
    """클래스별 속도 분포 박스플롯 (Plotly, 인터랙티브)."""
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
            boxpoints='outliers',   # 이상치만 개별 점으로, 겹침 고리 없음
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
```

- **주의 (재현 시 흔히 빠지는 함정)**:
  - matplotlib 차트 함수 내에서 `import numpy as np`와 `plt.cm.tab10`을 사용하던 코드는 Plotly 버전에서 완전히 삭제됨. `np`가 파일 다른 곳에서도 쓰인다면 상단 `import numpy as np`는 그대로 유지.
  - `make_subplots`의 `subplot_titles`에 한글을 넣어도 브라우저에서 정상 렌더링. matplotlib과 달리 폰트 설정 없음.
  - `legendgroup=lbl`과 `showlegend=False`를 두 번째 서브플롯 트레이스에 설정하는 것이 핵심 — 그렇지 않으면 같은 감지선 이름이 범례에 2번 나타남.
  - Plotly Figure를 반환해도 `gr.Plot` 컴포넌트는 그대로 사용 가능 (별도 컴포넌트 타입 변경 불필요).
  - `hovertemplate` 안의 `%{y}`, `%{x}` 는 Plotly 템플릿 변수이므로 Python f-string으로 만들면 안 됨. `<extra>...</extra>` 태그는 호버 상자의 오른쪽 파란 레이블(트레이스 이름) 을 커스텀하는 Plotly 문법.
- **파일**: `app.py`

### 11. 처리 결과가 숫자·차트뿐이라 패턴 해석은 전부 사용자가 직접 해야 함

- **증상**: "교통 분석 결과"/"도시 분석 결과"/"트랙 요약" 아코디언에 차트와 표는 풍부하지만, "이 영상에서 무슨 일이 있었는지"를 문장으로 설명해주는 기능이 없어 숫자를 사람이 직접 종합해야 했음.
- **원인**: 기존 코드에는 LLM/외부 API 호출이 전혀 없음(`requirements.txt`에 `anthropic`/`openai` 등 없음).
- **해결**: Claude API(Anthropic SDK)를 호출해 분석 결과를 한국어로 해설하는 기능을 추가. 매 처리마다 자동으로 비용이 발생하지 않도록 **별도 버튼**("AI 인사이트 생성")으로만 호출되며, API 키는 `.env`/환경변수(`ANTHROPIC_API_KEY`)에서만 읽고 UI나 세션 파일에는 절대 저장하지 않음.

**신규 파일 — `src/analysis/ai_insight.py`**: `last_analysis`/`last_result`에서 LLM에 보낼 작은 JSON 요약을 만드는 `build_analysis_summary()`와, Claude API를 호출하는 `generate_insight()`로 구성.

```python
# src/analysis/ai_insight.py
DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_CHOICES = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
MAX_TOKENS = 2048

class AIInsightError(Exception):
    """사용자에게 그대로 보여줄 수 있는 실패(키 누락/인증/한도/네트워크/거부)."""

def generate_insight(summary, model=DEFAULT_MODEL, api_key=None):
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIInsightError("ANTHROPIC_API_KEY가 설정되지 않았습니다. ...")
    import anthropic  # 지연 임포트 — 선택적 의존성으로 취급
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model, max_tokens=MAX_TOKENS,
        system=_build_system_prompt(summary),
        messages=[{"role": "user", "content": _build_user_message(summary)}],
    )
    # AuthenticationError/RateLimitError/APIStatusError/APIConnectionError →
    # 각각 한국어 AIInsightError로 변환, stop_reason == "refusal"과 빈 응답도 처리
    ...
```

**데이터 축소 전략** — 영상 길이에 비례해 무한정 늘어나는 원시 필드는 절대 그대로 보내지 않고, 이미 집계된 통계만 전달:

| 원본 필드 | 보내는 것 | 버리는 것 |
|---|---|---|
| `counting_lines` | `counts`, `flow_rates_veh_hr`, `headway`, `total_crossings`(계산) | `crossings`(통과마다 1건) |
| `speed` | `per_class`(클래스별 percentile 통계) | `track_speeds`, `track_cls`(트랙별 원시 샘플) |
| `od_matrix` | `matrix_df` → `{origin: {dest: count}}` | `raw`(튜플 키, JSON 불가) |
| `zone_analysis` | `zone_summaries` | `dwell_records`, `occupancy_timeseries`(초당 1건) |
| `track_summary` | 클래스별 집계(`count`, `mean_duration_frames`, `mean_distance`, `mean_speed_kmh`) | 트랙별 원시 행 전체(수백 개일 수 있음) |
| `congestion` | `events`(긴 순 상위 10개) + `total_events` | — |

`enabled_flags`(`enable_traffic`/`enable_speed`/`enable_od`/`enable_urban`)는 `last_analysis`에서 역추론하지 않고 `process_videos()`의 기존 로컬 변수를 그대로 전달 — 토글 OFF와 "토글 ON이지만 결과 0"은 `last_analysis` 모양만으로 구분 불가능하기 때문. `_build_system_prompt()`는 `summary`에 실제로 존재하는 키만 보고 리포트 섹션(교통 흐름/혼잡 구간/속도 분포/OD 흐름/존별 특이사항)을 동적으로 구성하며, "데이터에 없는 수치를 추측·생성하지 말 것"이라는 환각 방지 지침을 반드시 포함.

**`app.py` 변경 — `process_videos()` 출력 개수 15 → 16**:

```python
# process_videos() 끝, 최종 return 직전에 추가
enabled_flags = {
    "enable_traffic": bool(enable_traffic),
    "enable_speed": bool(enable_speed) and scale_mpp is not None,
    "enable_od": bool(enable_od),
    "enable_urban": bool(enable_urban),
}
ai_summary = build_analysis_summary(last_analysis, last_result, enabled_flags)
# return (..., batch_zip_path, ai_summary)   ← 맨 끝에 한 개 추가
```

출력 개수가 늘었으므로 아래 세 곳을 함께 수정해야 함:
- 조기 반환(`"영상을 먼저 업로드해주세요"`): `(None,) * 11` → `(None,) * 12`
- `_all_outputs` 리스트 끝에 `ai_insight_state`(새 `gr.State(value=None)`) 추가
- `video_input.clear(fn=lambda: (None,) * 15, ...)` → `(None,) * 16`

**`app.py` 변경 — UI 위젯 및 클릭 핸들러** ("트랙 요약" 아코디언 안, `track_summary_df_out` 바로 다음):

```python
with gr.Row():
    ai_model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL,
                                     label="Claude 모델", scale=2)
    ai_insight_btn = gr.Button("AI 인사이트 생성", variant="secondary", scale=1)
ai_insight_output = gr.Markdown(value="", elem_classes="note")
ai_insight_state = gr.State(value=None)
```

```python
def run_ai_insight(ai_summary, model):
    if not ai_summary:
        gr.Warning("먼저 영상을 처리해주세요. 분석 결과가 없습니다.")
        return gr.skip()
    try:
        return generate_insight(ai_summary, model=model)
    except AIInsightError as e:
        gr.Warning(str(e))
        return gr.skip()

ai_insight_btn.click(fn=run_ai_insight, inputs=[ai_insight_state, ai_model_dropdown],
                      outputs=[ai_insight_output])
```

`gr.Markdown`을 출력 컴포넌트로 쓴 이유는 Claude 응답의 `##` 헤더·불릿이 `gr.Textbox`에서는 `#`/`-` 글자 그대로 보이기 때문(Markdown은 실제 HTML로 렌더링). `gr.Warning` + `gr.skip()` 조합은 키 누락·인증 실패·한도 초과 등 예상 가능한 실패에서 토스트만 띄우고 이전 결과를 지우지 않기 위함 — `AIInsightError`가 아닌 예외(진짜 버그)는 잡지 않고 그대로 전파됨.

- **주의 (재현 시 흔히 빠지는 함정)**:
  - `ANTHROPIC_API_KEY`는 **반드시** `.env` 파일(`.gitignore`에 이미 포함됨) 또는 OS 환경변수로만 설정. `SessionConfig`(세션 JSON)에는 절대 추가하지 말 것 — 세션 파일을 공유하면 키가 새어나갈 수 있음. (※ 항목 12에서 UI 입력 필드가 추가되지만, 그 값은 브라우저 세션 동안만 유지되고 디스크에는 쓰지 않음 — 이 원칙은 유지됨)
  - `app.py` 상단에 `from dotenv import load_dotenv; load_dotenv()`를 다른 임포트보다 먼저 실행해야 `ANTHROPIC_API_KEY`가 `generate_insight()` 호출 시점에 채워져 있음.
  - 출력 개수 15 → 16 변경은 **세 곳**(최종 `return` 튜플, `_all_outputs` 리스트, `video_input.clear()`의 `(None,) * N`)을 모두 같이 고쳐야 함. 하나라도 빠뜨리면 Gradio가 "Number of output components does not match" 에러를 던짐. `restore_session()`의 `(gr.skip(),) * 68`은 입력 위젯만 다루므로 영향 없음.
  - `track_summary`는 트랙별 원시 행이 아니라 **클래스별 집계**(`_summarize_track_summary`)만 LLM에 전달됨 — 추적 객체가 수백 개여도 전송량은 클래스 수(보통 2~5개)만큼만 늘어남.
  - `anthropic` 패키지는 `generate_insight()` 내부에서 지연 임포트(`import anthropic`)됨 — 설치가 안 돼 있어도 앱 시작 자체는 깨지지 않고, 버튼을 눌렀을 때만 에러가 남.
  - `thinking`/`output_config.effort`는 의도적으로 사용하지 않음 — 이미 집계된 1~5KB JSON을 요약하는 단발성 작업이라 추론 단계가 품질에 도움이 되지 않고, 비용·지연만 늘어남.
- **파일**: `requirements.txt`, `src/analysis/ai_insight.py` (신규), `app.py`

### 12. AI 인사이트가 텍스트로만 분리돼 있고, Claude 계정 없는 사용자는 기능을 못 씀

- **증상 1 — 차트/표와 분리된 해설**: 항목 11에서 추가한 AI 인사이트는 단일 마크다운 텍스트 블록뿐이라, 어떤 데이터가 왜 주목할 만한지가 차트·표 위에서는 전혀 드러나지 않음.
- **증상 2 — Claude API 키 필수**: API 키가 없거나 Claude 계정이 없는 사용자는 기능 자체를 쓸 수 없음. `.env` 파일을 직접 편집해야 하는 것도 진입장벽.
- **해결**: ① Claude를 **구조화 출력**(`output_config.format` JSON 스키마)으로 호출해 `report_markdown`(기존과 동일한 전체 해설)과 `highlights`(차트·표에 표시할 메모 배열)를 함께 받음. ② API 키를 UI에서 직접 입력할 수 있는 비밀번호 필드 추가(세션 동안만 유지, `.env` 값은 폴백). ③ **로컬 Ollama**를 두 번째 AI 제공자로 추가해 키 없이도 기능을 쓸 수 있게 함(텍스트 리포트만, 하이라이트는 Claude 전용).

**`src/analysis/ai_insight.py` — `generate_insight()` 반환 타입 변경(str → dict)**:

```python
DEFAULT_OLLAMA_MODEL = "exaone3.5:7.8b"   # LG AI연구원, 한국어/영어 이중언어, 4.8GB
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
MAX_TOKENS = 3072  # 기존 2048 → highlights JSON 포함 여유분

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "report_markdown": {"type": "string"},
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": [
                        "counting_lines", "speed", "zone_analysis",
                        "od_matrix", "congestion", "track_summary",
                    ]},
                    "target": {"type": "string"},   # 데이터에 실제로 존재하는 라벨(클래스명/존 이름 등)
                    "note": {"type": "string"},
                    "importance": {"type": "string", "enum": ["high", "medium"]},
                },
                "required": ["category", "target", "note", "importance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["report_markdown", "highlights"],
    "additionalProperties": False,
}

def generate_insight(summary, provider="claude", *, model=DEFAULT_MODEL, api_key=None,
                      ollama_model=DEFAULT_OLLAMA_MODEL, ollama_host=DEFAULT_OLLAMA_HOST) -> dict:
    if provider == "ollama":
        return _generate_with_ollama(summary, ollama_model, ollama_host)
    return _generate_with_claude(summary, model, api_key)
```

Claude 경로는 `output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}`을 붙여 응답이 항상 스키마를 만족하는 JSON임을 보장받고 `json.loads()`만 함. Ollama 경로(`httpx.post(f"{host}/api/chat", json={...}, timeout=120.0)`)는 JSON 강제 없이 같은 섹션 구성의 마크다운 텍스트만 요청하고 `highlights`는 항상 `[]`로 채워 반환 모양을 통일함(`anthropic`의 전이 의존성으로 이미 설치돼 있던 `httpx`를 그대로 재사용 — 새 의존성 없음).

**`app.py` — 이미 그린 Figure/DataFrame에 AI 메모를 덧붙이는 두 헬퍼**(차트/표 본체 함수는 손대지 않음):

```python
def _annotate_chart(fig, highlights, category, xref="x", yref="y domain"):
    if fig is None or not highlights:
        return fig
    targets = [h for h in highlights if h.get("category") == category]
    if not targets:
        return fig
    fig = copy.deepcopy(fig)   # 원본은 그대로 두고 복사본에만 주석 추가 — 재생성 시 누적 방지
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
        color = "#fef3c7" if h.get("importance") == "high" else "#eff6ff"
        return [f"background-color: {color}"] * len(row)
    return df.style.apply(_row_style, axis=1)
```
통과수/유량 차트는 서브플롯 1("클래스별 통과 수")에만 주석을 단다(서브플롯 2와 x축 카테고리가 같아 중복 표시할 필요 없음). `gr.DataFrame`(Gradio 6.14.0 확인)은 pandas `Styler` 객체를 그대로 받아 셀 배경색을 렌더링하지만, **`interactive=False`일 때만** 동작하므로 `zone_df_out`/`track_summary_df_out` 선언에 `interactive=False`를 명시적으로 추가했음. OD 행렬은 행×열 쌍 매칭이 필요해 복잡도가 커서 v1에서는 제외(텍스트 강조점에는 포함됨).

**`process_videos()` — 출력 개수는 그대로 16개**: 차트를 다시 그리려면 원본 Figure/DataFrame이 필요한데, 이를 새 `gr.State` 슬롯으로 추가하지 않고 **기존 `ai_insight_state`에 저장하는 값의 모양만 dict로 확장**했음(`{"llm_summary":..., "traffic_count_fig":..., "traffic_speed_fig":..., "zone_df":..., "track_summary_df":...}`). `track_summary_df`는 기존에 `return` 문 안에서 인라인으로 한 번 더 만들던 것을 변수로 추출해 한 번만 생성하도록 바꿈.

**UI — 제공자 라디오 + 조건부 그룹**:
```python
ai_provider_radio = gr.Radio(choices=["Claude API", "로컬 (Ollama)"], value="Claude API", label="AI 제공자")
with gr.Group(visible=True) as claude_group:
    ai_model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Claude 모델")
    ai_api_key_box = gr.Textbox(label="Anthropic API 키 (선택)", type="password",
                                 placeholder="sk-ant-... (비워두면 .env의 ANTHROPIC_API_KEY 사용)")
with gr.Group(visible=False) as ollama_group:
    ai_ollama_model_box = gr.Textbox(value=DEFAULT_OLLAMA_MODEL, label="Ollama 모델")
    ai_ollama_host_box = gr.Textbox(value=DEFAULT_OLLAMA_HOST, label="Ollama 주소")

ai_provider_radio.change(
    fn=lambda p: (gr.update(visible=(p == "Claude API")), gr.update(visible=(p == "로컬 (Ollama)"))),
    inputs=[ai_provider_radio], outputs=[claude_group, ollama_group],
)
```
`run_ai_insight()`의 입력이 1개(모델명)에서 6개(스냅샷·제공자·Claude 모델·API 키·Ollama 모델·Ollama 주소)로, 출력이 1개(텍스트)에서 5개(텍스트 + 통과수 차트 + 속도 차트 + 존 표 + 트랙 표)로 늘어남. 실패 시(`AIInsightError`, 또는 스냅샷 없음) 5개 출력 모두 `gr.skip()`을 반환해 이전 결과를 보존.

- **주의 (재현 시 흔히 빠지는 함정)**:
  - `interactive=False`를 빠뜨리면 Gradio가 Styler를 무시하고 일반 값으로 렌더링해 배경색이 안 보일 수 있음 — `zone_df_out`/`track_summary_df_out` 둘 다 확인.
  - `_annotate_chart`/`_highlight_dataframe`는 항상 `ai_insight_state`에 저장된 **원본**(`process_videos()`가 만든 깨끗한 버전)에서 시작해야 함. `run_ai_insight()`가 반환한 이전 결과(이미 주석이 붙은 Figure)를 다시 입력으로 쓰면 안 됨 — 그래서 스냅샷을 따로 보관함.
  - 로컬 모델은 `highlights`가 항상 `[]`이므로 `render_insight_markdown()`이 자동으로 "🔍 주목할 포인트" 블록 없이 `report_markdown`만 반환함 — 별도 분기 불필요.
  - Ollama 모델 태그는 정확히 일치해야 함(`exaone3.5:7.8b`처럼 크기까지 포함). 모델을 `ollama pull` 하지 않은 상태로 호출하면 404가 와서 "pull 실행" 안내 경고가 뜸.
  - UI의 API 키 입력란은 비워두면 `.env`/환경변수로 자동 폴백하고, 값을 입력하면 그 값이 우선됨 — `generate_insight(..., api_key=(claude_api_key or None))`로 빈 문자열을 `None`으로 변환해야 폴백이 정상 동작함.
- **파일**: `requirements.txt`, `src/analysis/ai_insight.py`, `app.py`

#### 후속 수정 — 다크 테마에서 하이라이트 행 글자 안 보임 + AI 인사이트가 "트랙 요약"에 묻혀 있던 문제

> 항목 12 배포 직후 실제 화면에서 두 가지 문제가 발견되어 수정.

- **증상 1 — 하이라이트 행이 하얗게 칠해져 데이터가 안 보임**: "존 분석"/"트랙별 요약" 표에서 AI가 강조한 행이 흰색(또는 거의 흰색)으로 덮여 그 행의 모든 글자가 보이지 않음.
- **증상 2 — AI 인사이트가 별도 영역이 아님**: AI 인사이트 전체 해설을 "트랙 요약" 아코디언을 열어야만 볼 수 있어, 두 기능이 한 섹션에 묻혀 있었음.
- **원인 1**: `_highlight_dataframe`의 `_row_style`이 `background-color`만 지정하고 글자색은 지정하지 않음. 앱이 다크 테마라 기본 글자색이 밝은색인데, 하이라이트 배경도 밝은 파스텔색(`#fef3c7`, `#eff6ff`)이라 밝은 글자 + 밝은 배경 = 글자가 사실상 보이지 않게 됨.
- **원인 2**: "AI 인사이트" UI 블록 전체(제공자 라디오, 모델/키 입력, 버튼, 출력 패널)를 `with gr.Accordion("트랙 요약", ...)` 블록 안에 넣어서 별도 아코디언으로 분리하지 않았음.
- **해결 1**: `_row_style`이 반환하는 CSS 문자열에 어두운 글자색을 명시적으로 고정:

```python
# app.py — _highlight_dataframe 내부 _row_style
def _row_style(row):
    h = targets.get(row[match_column])
    if not h:
        return [""] * len(row)
    bg = "#fef3c7" if h.get("importance") == "high" else "#eff6ff"
    # 다크 테마에서도 글자가 보이도록 글자색을 명시적으로 어둡게 고정
    return [f"background-color: {bg}; color: #1a1a1a"] * len(row)
```

- **해결 2**: "트랙 요약" 아코디언에서 AI 인사이트 관련 위젯들을 모두 빼내 `with gr.Accordion("AI 인사이트", open=True):`라는 별도 아코디언으로 분리(컴포넌트 변수명·`run_ai_insight()` 클릭 핸들러 wiring은 그대로 — 레이아웃 위치만 이동). 버튼도 `variant="secondary"` → `variant="primary"`로 바꿔 독립된 기능임을 시각적으로 강조:

```python
with gr.Accordion("트랙 요약", open=True):
    ...
    track_summary_df_out = gr.DataFrame(label="트랙별 요약", interactive=False)

with gr.Accordion("AI 인사이트", open=True):   # ← 새 별도 아코디언
    gr.Markdown("Claude API 또는 로컬 AI(Ollama)로 위 분석 결과를 한국어로 해설합니다. ...")
    ai_provider_radio = gr.Radio(...)
    with gr.Group(visible=True) as claude_group: ...
    with gr.Group(visible=False) as ollama_group: ...
    ai_insight_btn = gr.Button("AI 인사이트 생성", variant="primary")
    ai_insight_output = gr.Markdown(value="", elem_classes="note")
    ai_insight_state = gr.State(value=None)
```

- **주의 (재현 시 흔히 빠지는 함정)**:
  - 차트 주석(`_annotate_chart`)은 이 문제와 무관함 — 배지 배경이 이미 어두운 채도색(`#b45309`, `#1d4ed8`)이고 글자색이 흰색으로 고정돼 있어 테마와 무관하게 항상 대비가 충분함. 문제는 **표** 하이라이트(밝은 파스텔 배경)에만 있었음.
  - pandas Styler의 CSS 문자열은 세미콜론으로 여러 속성을 이어 쓸 수 있음(`"background-color: X; color: Y"`) — 한 속성만 덮어쓰면 나머지는 테마 기본값을 그대로 물려받는다는 점을 기억할 것.
  - 아코디언 분리는 순수 레이아웃 변경이라 `run_ai_insight()`나 `.click()` 와이어링, `process_videos()`의 16개 출력 개수에는 영향 없음 — 컴포넌트가 Python 변수로 참조되는 한 어느 `with gr.Accordion(...)` 블록 안에 있는지는 무관함.
- **파일**: `app.py`

---

### 13. OD 행렬이 항상 비어 있고, 설정 방법을 화면에서 찾을 수 없음

- **증상**: "OD 행렬" 체크박스를 켜고 처리해도 결과 표가 항상 비어 있거나 표시되지 않음. 무엇을 더 설정해야 하는지 화면 어디에도 안내가 없었음.
- **원인 1 (숨겨진 의존성 버그)**: `_build_analyzers()`가 `enable_od`와 `zones`뿐 아니라 **`enable_traffic`까지 동시에 켜져 있어야** `ODMatrixBuilder`를 만들었음(`src/pipeline.py`). OD 행렬은 본질적으로 "존" 기반 기능이라 가상 감지선(교통 분석)과는 무관한데, 같은 아코디언에 체크박스가 있다는 이유만으로 강제 연동되어 있었음.
- **원인 2 (발견 불가능한 UI 배치)**: "OD 행렬" 체크박스는 "교통 분석" 아코디언에 있었지만, 그 체크박스가 실제로 필요로 하는 "존(zone)" 설정은 완전히 다른 아코디언인 "도시 공간 분석"에 있었음. 두 설정이 서로 다른 섹션에 분리돼 있어 사용자가 둘 다 켜야 한다는 사실을 알 방법이 없었음.
- **해결 1**: `enable_traffic` 조건 제거 — OD 행렬은 `존` + `enable_od`만으로 독립 동작하도록 수정:

```python
# src/pipeline.py — _build_analyzers()
zones = analysis_config.get('zones', [])
if zones and analysis_config.get('enable_od'):      # enable_traffic 조건 삭제
    from .analysis.od_matrix import ODMatrixBuilder
    analyzers.append(ODMatrixBuilder(zones))
```

- **해결 2**: "OD 행렬 계산" 체크박스를 "교통 분석"에서 빼내 실제 의존 대상인 "도시 공간 분석"의 존 정의 바로 아래로 이동하고, "최소 2개 이상의 존 필요"·"도시 분석 활성화와 무관하게 독립 동작" 안내문을 함께 추가.
- **해결 3 — 빈 결과에 대한 안내**: 결과가 비어 있는 상황을 더 이상 빈 표로 침묵 처리하지 않고, 원인을 구분해 안내 메시지를 표시:

```python
# app.py — process_videos()
elif enable_od:
    if len(zones) < 2:
        od_msg = "OD 행렬에는 최소 2개 이상의 존이 필요합니다. '도시 공간 분석' 아코디언에서 존을 2개 이상 활성화하고 좌표를 입력해주세요."
    else:
        od_msg = "이번 영상에서는 활성화된 존 사이를 이동한 객체가 감지되지 않았습니다."
    od_df = pd.DataFrame({"안내": [od_msg]})
```

- **부가 수정 — AI 인사이트 버튼 중복 클릭 방지**: 생성 중에도 버튼이 계속 클릭 가능해 같은 요청이 중첩 발생할 수 있었음. `.click()`을 3단계 체인(`버튼 비활성화 → run_ai_insight → 버튼 재활성화`)으로 분리해 생성 중에는 버튼이 "AI 인사이트 생성 중..."으로 바뀌고 비활성화됨:

```python
ai_insight_btn.click(
    fn=lambda: gr.update(interactive=False, value="AI 인사이트 생성 중..."),
    outputs=[ai_insight_btn],
).then(
    fn=run_ai_insight, inputs=[...], outputs=[...],
).then(
    fn=lambda: gr.update(interactive=True, value="AI 인사이트 생성"),
    outputs=[ai_insight_btn],
)
```
`run_ai_insight()`는 실패 시에도 예외를 던지지 않고 `(gr.skip(),) * 5`를 반환하므로, 체인의 마지막 단계(재활성화)는 성공/실패 모두에서 항상 실행됨.
- **주의 (재현 시 흔히 빠지는 함정)**:
  - `enable_od` 체크박스를 옮겨도 `process_videos()`의 매개변수 순서나 `inputs=[...]` 리스트는 그대로 둬도 됨 — Gradio 와이어링은 컴포넌트가 어느 `with gr.Accordion(...)` 블록에 있는지가 아니라 Python 변수 자체를 참조하므로, UI상 위치 이동은 함수 시그니처/입력 리스트와 무관함.
  - 존(zone)은 "도시 분석 활성화"(`enable_urban`) 체크박스와 무관하게 항상 정의 가능 — `enable_urban`은 `ZoneAnalyzer`/히트맵에만 영향을 주고, OD 행렬은 별도로 `zones` 리스트 자체만 본다.
  - OD 행렬이 0이 아닌 값을 가지려면 같은 트랙이 서로 다른 두 존을 실제로 통과해야 함 — 존 1개만 활성화하면 모든 진입이 "같은 곳"이라 거리(전이)가 기록되지 않아 항상 빈 결과가 나옴(버그 아님, 정상 동작).
- **파일**: `src/pipeline.py`, `app.py`

---

### 14. 존을 2개 이상 설정해도 OD 행렬이 항상 0으로만 나옴

> 항목 13으로 설정/UI 문제를 고친 뒤에도, 실제 화면에서 두 존(Zone A/B)에 모두 진입 기록이 있는데 OD 행렬 결과가 여전히 전부 0으로 나오는 문제가 보고됨 — 항목 13과는 별개의, 더 근본적인 로직 버그.

- **증상**: "Zone A [5]", "Zone B [1]"처럼 두 존 모두에 진입 기록이 있음에도, OD 행렬 표는 모든 칸이 0으로 표시됨.
- **원인**: `ODMatrixBuilder.update()`(`src/analysis/od_matrix.py`)가 트랙의 "이전 존"을 **바로 직전 프레임의 존**으로만 기억하고 있었음. 두 존 사이에 존이 아닌 일반 도로 구간이 있으면(현실에서는 거의 항상 그러함), 그 구간을 지나는 프레임마다 `_track_in_zone[tid] = None`으로 덮어써져 "Zone A에 있었다"는 기억이 사라짐. 그 결과 다음 존(Zone B)에 진입한 순간엔 `prev`가 이미 `None`이라 전이가 절대 기록되지 않음 — 두 존이 서로 붙어 있어 한 프레임 안에 바로 옮겨가는 경우에만 정상 작동하는 구조였음.
- **해결**: 트랙이 어떤 존에도 속하지 않는 프레임은 그냥 건너뛰도록 변경 — `_track_in_zone[tid]`는 트랙이 **마지막으로 확인된 존**만 기억하고, 존이 아닌 구간을 지나도 지워지지 않음:

```python
# src/analysis/od_matrix.py — ODMatrixBuilder.update()
def update(self, frame_idx, track_data, fps):
    for td in track_data:
        tid = td['track_id']
        cx, cy = td['cx'], td['cy']
        zone = self._get_zone(cx, cy)
        if zone is None:
            continue   # 존 밖 — 마지막으로 확인된 존 기억을 지우지 않음
        prev = self._track_in_zone.get(tid)
        if tid not in self._track_origin:
            self._track_origin[tid] = zone
        if prev is not None and prev != zone:
            origin = self._track_origin.get(tid, prev)
            if origin != zone:
                self.matrix[(origin, zone)] = self.matrix.get((origin, zone), 0) + 1
        self._track_in_zone[tid] = zone
```

검증: 트랙이 Zone A(연속 2프레임) → 존 없는 구간(연속 3프레임) → Zone B로 이동하는 시나리오를 시뮬레이션해, 수정 전엔 전이가 0건 기록되고 수정 후엔 `(Zone A, Zone B): 1`이 정확히 기록됨을 확인.

- **주의 (재현 시 흔히 빠지는 함정)**:
  - `origin`은 트랙이 **최초로** 진입한 존으로 고정되고 이후 다른 존에 들어가도 갱신되지 않음(의도된 동작) — 즉 A→B→C로 이동하면 (A,B)와 (A,C) 둘 다 기록되고 (B,C)는 기록되지 않음. "최초 출발지 기준" OD 행렬이라는 설계를 유지했고, 이번 수정은 그 설계 자체가 아니라 "존 사이 공백 구간에서 기억이 끊기는" 버그만 고친 것.
  - 같은 존 안에 머무는 동안은(`prev == zone`) 매 프레임 중복 카운트되지 않음 — 이 부분은 기존에도 정상이었음.
- **파일**: `src/analysis/od_matrix.py`

---

## 라이선스

MIT
