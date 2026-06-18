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

---

### 11. 교통 분석 차트에 한글이 깨지고, 속도 박스플롯에 고리 아티팩트가 발생하며, 값을 정확히 읽기 어려움

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

---

## 라이선스

MIT
