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
>
> **정리 원칙**: 같은 기능이 여러 차례 수정된 경우(최초 구현 → 추가 버그 발견 → 재수정), 각 시도를 별도 항목으로 나열하지 않고 **하나의 항목에 최종 상태만** 기록합니다. 중간에 폐기된 코드(예: 나중에 다른 구현으로 전면 교체된 버전)는 보여주지 않고, 무엇이 왜 바뀌었는지만 짧게 요약합니다 — 그래야 이 문서를 따라가는 코딩 에이전트가 이미 틀린 것으로 판명된 중간 버전을 굳이 구현했다가 다시 고치는 일이 없습니다. 다만 "원인이 하나가 아니었다"는 사실 자체는 디버깅 맥락으로 유용하므로, 그런 경우는 한 항목 안에 "원인 1/2/3"처럼 모아 적습니다. 각 항목의 **출력 개수(`process_videos`의 반환 튜플 길이 등)는 항목을 순서대로 적용했을 때의 누적 값**이므로, 중간 항목을 건너뛰면 개수가 맞지 않습니다.

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

- **증상**: 처리 후 "교통 분석 결과" 아코디언을 열면 숫자가 채워진 DataFrame 테이블(감지선·클래스·통과수·유량·차두시간·속도 등)만 표시됨. 여러 클래스 간 통과 수 크기 차이, 속도 분포의 이상치 유무 같은 정보를 눈으로 파악하려면 숫자를 하나씩 읽어야 했음.
- **원인**: `process_videos()`가 `traffic_df`, `speed_df`를 `pd.DataFrame`으로 만들어 `gr.DataFrame` 컴포넌트에 반환. 결과 화면에서 차트를 렌더링하는 코드가 없었음.
- **해결**: 두 개의 차트 생성 함수를 추가하고 `gr.DataFrame` 출력을 `gr.Plot` 출력 두 개로 교체.

> **이력**: 최초에는 matplotlib으로 구현했으나, 실제 화면에서 ① 한글이 모두 □□□ 박스 문자로 깨짐(시스템에 한글 폰트를 못 찾음), ② 속도 박스플롯에서 이상치 마커가 겹쳐 고리·도넛 모양으로 보임, ③ 정적 PNG라 호버 시 정확한 수치가 안 나옴 — 세 가지 문제가 발견되어 **즉시 Plotly로 전면 교체**했음. 아래는 그 최종(Plotly) 버전만 기록한다.

**함수 1 — 통과 수 · 유량 막대 차트**:

```python
# app.py — _resolve_file_paths() 이후에 추가
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
```

**함수 2 — 속도 분포 박스플롯**:

```python
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
            boxpoints='outliers',   # 이상치만 개별 점으로 표시 — 겹침 고리 없음
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
  - Plotly Figure를 반환해도 `gr.Plot` 컴포넌트는 그대로 사용 가능(matplotlib Figure와 Plotly Figure를 모두 지원) — 컴포넌트 타입 변경은 필요 없음. `make_subplots`의 `subplot_titles`에 한글을 넣어도 브라우저가 직접 렌더링하므로 폰트 설정이 전혀 필요 없음.
  - `legendgroup=lbl`과 `showlegend=False`를 두 번째 서브플롯 트레이스에 설정하는 것이 핵심 — 빠뜨리면 같은 감지선 이름이 범례에 2번 나타남.
  - `hovertemplate` 안의 `%{y}`, `%{x}`는 Plotly 템플릿 변수이므로 Python f-string으로 만들면 안 됨. `<extra>...</extra>` 태그는 호버 상자의 오른쪽 파란 레이블(트레이스 이름)을 커스텀하는 Plotly 문법.
  - 반환 타입이 `None`인 경우(`counting_data`가 비어있는 등)에도 `gr.Plot`은 조용히 빈 상태로 표시됨 — 별도 처리 불필요.
  - 출력 개수가 14 → 15개로 늘었으므로, 조기 반환 경로(`"영상을 먼저 업로드해주세요"`)와 `video_input.clear()`의 자리 채우기도 같이 늘려야 함:
    - 조기 반환: `(None, None, None, "영상을 먼저 업로드해주세요.") + (None,) * 11`  (10 → 11)
    - clear: `fn=lambda: (None,) * 15`  (14 → 15)
- **파일**: `app.py`

---

### 11. 처리 결과가 숫자·차트뿐이라 패턴 해석은 전부 사용자가 직접 해야 함

- **증상**: "교통 분석 결과"/"도시 분석 결과"/"트랙 요약" 아코디언에 차트와 표는 풍부하지만, "이 영상에서 무슨 일이 있었는지"를 문장으로 설명해주는 기능이 없어 숫자를 사람이 직접 종합해야 했음. 또한 API 키가 없거나 Claude 계정이 없는 사용자는 기능 자체를 못 쓰고, 해설이 차트·표와 분리된 텍스트 블록뿐이라 어떤 데이터가 왜 주목할 만한지 시각적으로 드러나지 않았음.
- **원인**: 기존 코드에는 LLM/외부 API 호출이 전혀 없음(`requirements.txt`에 `anthropic` 등 없음).
- **해결**: Claude API(Anthropic SDK)를 호출해 분석 결과를 한국어로 해설하는 **AI 인사이트** 기능을 추가. 매 처리마다 자동으로 비용이 발생하지 않도록 **별도 버튼**으로만 호출하고, ① Claude를 구조화 출력으로 호출해 해설 텍스트와 "차트·표에 표시할 메모(하이라이트)"를 함께 받고, ② API 키를 UI에서 직접 입력 가능하게 하며(세션 동안만 유지, `.env` 값은 폴백), ③ 키 없이도 쓸 수 있는 **로컬 Ollama**를 두 번째 제공자로 추가.

> **이력**: 이 기능은 한 번에 지금 형태로 만들어지지 않았다. ① 최초 버전은 Claude 단일 제공자 + 일반 텍스트 반환 + 단일 클릭 버튼이었고, ② 이후 Claude 구조화 출력(`highlights`) + 로컬 Ollama 제공자 + UI API 키 입력으로 확장되었고, ③ 배포 직후 다크 테마에서 하이라이트 행 글자가 안 보이는 버그와 AI 인사이트가 "트랙 요약" 아코디언에 묻혀 있던 레이아웃 문제가 발견되어 즉시 수정되었고, ④ 생성 중 버튼을 다시 누르면 요청이 중첩되는 문제가 발견되어 버튼을 일시 비활성화하는 처리가 추가되었다. 아래는 그 모든 과정을 거친 **최종 코드만** 기록한다.

**신규 파일 — `src/analysis/ai_insight.py`**: `last_analysis`/`last_result`에서 LLM에 보낼 작은 JSON 요약을 만드는 `build_analysis_summary()`와, LLM을 호출하는 `generate_insight()`(provider에 따라 `_generate_with_claude`/`_generate_with_ollama`로 분기)로 구성.

```python
# src/analysis/ai_insight.py
DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_CHOICES = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
MAX_TOKENS = 3072   # report_markdown + highlights JSON을 함께 받으므로 여유 있게 설정

DEFAULT_OLLAMA_MODEL = "exaone3.5:7.8b"   # LG AI연구원, 한국어/영어 이중언어, 4.8GB
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

_CATEGORY_LABELS = {
    "counting_lines": "교통", "speed": "속도", "zone_analysis": "존",
    "od_matrix": "OD", "congestion": "혼잡", "track_summary": "트랙",
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "report_markdown": {"type": "string"},
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(_CATEGORY_LABELS)},
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


class AIInsightError(Exception):
    """사용자에게 그대로 보여줄 수 있는 실패(키 누락/인증/한도/네트워크/거부).

    app.py는 이 예외만 gr.Warning으로 변환하고, 그 외 예외는 버그로 취급해 그대로 전파한다.
    """


def generate_insight(summary, provider="claude", *, model=DEFAULT_MODEL, api_key=None,
                      ollama_model=DEFAULT_OLLAMA_MODEL, ollama_host=DEFAULT_OLLAMA_HOST) -> dict:
    """provider="claude"면 구조화 출력으로 {"report_markdown", "highlights"}를 받고,
    provider="ollama"면 로컬 서버에서 텍스트 리포트만 받는다(highlights는 항상 []).
    두 경로 모두 같은 모양의 dict를 반환한다."""
    if provider == "ollama":
        return _generate_with_ollama(summary, ollama_model, ollama_host)
    return _generate_with_claude(summary, model, api_key)


def _generate_with_claude(summary, model, api_key) -> dict:
    import os
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIInsightError(
            "Anthropic API 키가 없습니다. 위 'Anthropic API 키' 입력란에 붙여넣거나 "
            ".env 파일에 ANTHROPIC_API_KEY=sk-ant-... 를 추가한 뒤 다시 시도해주세요."
        )
    import anthropic  # 지연 임포트: 선택적 의존성으로 취급 — 미설치 시 앱 시작은 깨지지 않음
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            system=_build_structured_system_prompt(summary),
            messages=[{"role": "user", "content": _build_user_message(summary)}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
    except anthropic.AuthenticationError:
        raise AIInsightError("Claude API 인증에 실패했습니다. API 키가 올바른지 확인해주세요.")
    except anthropic.RateLimitError:
        raise AIInsightError("Claude API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except anthropic.APIStatusError as e:
        raise AIInsightError(f"Claude API 오류가 발생했습니다 (status={e.status_code}).")
    except anthropic.APIConnectionError:
        raise AIInsightError("Claude API 서버에 연결할 수 없습니다. 네트워크를 확인해주세요.")

    if response.stop_reason == "refusal":
        raise AIInsightError("Claude가 이 요청에 대한 응답을 거부했습니다.")
    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise AIInsightError("Claude로부터 빈 응답을 받았습니다.")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise AIInsightError("Claude 응답을 해석할 수 없습니다. 다시 시도해주세요.")
    result.setdefault("highlights", [])
    return result


def _generate_with_ollama(summary, model, host) -> dict:
    import httpx  # anthropic의 전이 의존성으로 이미 설치돼 있음 — 새 의존성 없음
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_text_system_prompt(summary)},  # JSON 강제 없는 일반 텍스트 프롬프트
            {"role": "user", "content": _build_user_message(summary)},
        ],
        "stream": False,
    }
    try:
        resp = httpx.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=120.0)
    except httpx.ConnectError:
        raise AIInsightError(f"로컬 Ollama 서버({host})에 연결할 수 없습니다. Ollama를 설치·실행한 뒤 다시 시도해주세요.")
    except httpx.TimeoutException:
        raise AIInsightError("Ollama 응답 시간이 초과되었습니다. 더 작은 모델을 사용해보세요.")

    if resp.status_code == 404:
        raise AIInsightError(f"Ollama 모델 '{model}'을 찾을 수 없습니다. `ollama pull {model}` 실행 후 다시 시도해주세요.")
    if resp.status_code != 200:
        raise AIInsightError(f"Ollama 오류가 발생했습니다 (status={resp.status_code}).")

    text = (resp.json().get("message") or {}).get("content", "").strip()
    if not text:
        raise AIInsightError("Ollama로부터 빈 응답을 받았습니다.")
    return {"report_markdown": text, "highlights": []}   # highlights는 항상 빈 리스트 — Claude 전용


def render_insight_markdown(result: dict) -> str:
    """highlights가 있으면(Claude 경로) 맨 위에 강조점 블록을 추가하고, 없으면(Ollama 경로,
    또는 Claude가 하이라이트를 찾지 못한 경우) report_markdown만 그대로 반환한다."""
    highlights = result.get("highlights") or []
    report = result.get("report_markdown", "")
    if not highlights:
        return report
    lines = ["## 🔍 주목할 포인트 (연구 방향성 힌트)", ""]
    for h in sorted(highlights, key=lambda h: 0 if h.get("importance") == "high" else 1):
        mark = "🔴" if h.get("importance") == "high" else "🔵"
        label = _CATEGORY_LABELS.get(h.get("category"), h.get("category", ""))
        lines.append(f"- {mark} **[{label}] {h.get('target', '')}** — {h.get('note', '')}")
    lines.append("\n---\n")
    return "\n".join(lines) + report
```

`_build_structured_system_prompt(summary)`/`_build_text_system_prompt(summary)`는 `summary`에 실제로 존재하는 키만 보고 리포트 섹션(교통 흐름/혼잡 구간/속도 분포/OD 흐름/존별 특이사항/트랙 요약/권장사항)을 동적으로 구성하며, "데이터에 없는 수치를 추측·생성하지 말 것"이라는 환각 방지 지침을 반드시 포함한다. Claude용 프롬프트에는 추가로 "highlights 배열에는 데이터에 실제로 존재하는 라벨만 정확히 사용해 가장 주목할 만한 데이터 포인트 3~6개를 뽑으라"는 지침이 붙는다. 두 프롬프트 모두 `_build_user_message(summary)`(요약 dict를 `json.dumps`)를 사용자 메시지로 보낸다.

**데이터 축소 전략** — 영상 길이에 비례해 무한정 늘어나는 원시 필드는 절대 그대로 보내지 않고, 이미 집계된 통계만 전달:

| 원본 필드 | 보내는 것 | 버리는 것 |
|---|---|---|
| `counting_lines` | `counts`, `flow_rates_veh_hr`, `headway`, `total_crossings`(계산) | `crossings`(통과마다 1건) |
| `speed` | `per_class`(클래스별 percentile 통계) | `track_speeds`, `track_cls`(트랙별 원시 샘플) |
| `od_matrix` | `matrix_df` → `{origin: {dest: count}}` | `raw`(튜플 키, JSON 불가) |
| `zone_analysis` | `zone_summaries` | `dwell_records`, `occupancy_timeseries`(초당 1건) |
| `track_summary` | 클래스별 집계(`count`, `mean_duration_frames`, `mean_distance`, `mean_speed_kmh`) | 트랙별 원시 행 전체(수백 개일 수 있음) |
| `congestion` | `events`(긴 순 상위 10개) + `total_events` | — |

`enabled_flags`(`enable_traffic`/`enable_speed`/`enable_od`/`enable_urban`)는 `last_analysis`에서 역추론하지 않고 `process_videos()`의 기존 로컬 변수를 그대로 전달 — 토글 OFF와 "토글 ON이지만 결과 0"은 `last_analysis` 모양만으로 구분 불가능하기 때문.

**`app.py` — 이미 그린 Figure/DataFrame에 AI 메모를 덧붙이는 두 헬퍼**(차트/표 본체 함수는 손대지 않음):

```python
import copy   # app.py 상단에 추가

def _annotate_chart(fig, highlights, category, xref="x", yref="y domain"):
    """원본 fig는 건드리지 않고 깊은 복사본에 주석을 추가해 반환 — 재생성할 때마다
    이전 주석이 누적되지 않도록 항상 process_videos()가 만든 원본에서 다시 그린다."""
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
    """category에 해당하는 하이라이트를 df에 "AI 메모" 컬럼으로 덧붙이고, 일치하는 행에
    배경색을 입힌 pandas Styler를 반환한다."""
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
        # 다크 테마 기본 글자색(밝은색)과 밝은 파스텔 배경이 겹쳐 글자가 안 보이는 문제가 있었으므로,
        # 배경색만 지정하지 말고 글자색도 항상 함께 명시적으로 고정한다.
        return [f"background-color: {bg}; color: #1a1a1a"] * len(row)

    return df.style.apply(_row_style, axis=1)
```
통과수/유량 차트는 서브플롯 1("클래스별 통과 수")에만 주석을 단다(서브플롯 2와 x축 카테고리가 같아 중복 표시할 필요 없음). `gr.DataFrame`(Gradio 6.14.0 확인)은 pandas `Styler` 객체를 그대로 받아 셀 배경색을 렌더링하지만, **`interactive=False`일 때만** 동작하므로 `zone_df_out`/`track_summary_df_out` 선언에 `interactive=False`를 명시적으로 추가해야 함. 차트 배지(`_annotate_chart`)는 배경이 이미 어두운 채도색(`#b45309`, `#1d4ed8`)이고 글자색이 흰색으로 고정돼 있어 테마와 무관하게 항상 대비가 충분하므로 같은 문제가 없음 — 다크 테마 글자색 문제는 **표** 하이라이트에만 있었음. OD 행렬은 행×열 쌍 매칭이 필요해 복잡도가 커서 차트/표 하이라이트 대상에서는 제외(텍스트 강조점에는 포함됨).

**`app.py` — `process_videos()` 출력 개수 15 → 16**: 차트를 다시 그리려면 원본 Figure/DataFrame이 필요하므로, `ai_summary` 하나만이 아니라 그걸 포함한 스냅샷 dict를 새 `gr.State`(`ai_insight_state`)에 저장한다:

```python
# process_videos() 끝, 최종 return 직전에 추가
enabled_flags = {
    "enable_traffic": bool(enable_traffic),
    "enable_speed": bool(enable_speed) and scale_mpp is not None,
    "enable_od": bool(enable_od),
    "enable_urban": bool(enable_urban),
}
ai_summary = build_analysis_summary(last_analysis, last_result, enabled_flags)
ai_insight_snapshot = {
    "llm_summary": ai_summary,
    "traffic_count_fig": traffic_count_fig,
    "traffic_speed_fig": traffic_speed_fig,
    "zone_df": zone_df,
    "track_summary_df": track_summary_df,
}
# return (..., batch_zip_path, ai_insight_snapshot)   ← 맨 끝에 한 개 추가
```

출력 개수가 늘었으므로 아래 세 곳을 함께 수정해야 함(하나라도 빠뜨리면 Gradio가 "Number of output components does not match" 에러를 던짐. `restore_session()`의 `(gr.skip(),) * 68`은 입력 위젯만 다루므로 영향 없음):
- 조기 반환(`"영상을 먼저 업로드해주세요"`): `(None,) * 11` → `(None,) * 12`
- `_all_outputs` 리스트 끝에 `ai_insight_state` 추가
- `video_input.clear(fn=lambda: (None,) * 15, ...)` → `(None,) * 16`

**`app.py` — UI**: "트랙 요약"과는 별도로 독립된 `"AI 인사이트"` 아코디언을 만들어 제공자 선택 라디오 + 조건부 입력 그룹을 둔다(같은 아코디언 안에 섞으면 기능이 묻혀서 찾기 어려워짐):

```python
with gr.Accordion("트랙 요약", open=True):
    ...
    track_summary_df_out = gr.DataFrame(label="트랙별 요약", interactive=False)

with gr.Accordion("AI 인사이트", open=True):
    gr.Markdown(
        "Claude API 또는 로컬 AI(Ollama)로 위 분석 결과를 한국어로 해설합니다. "
        "주목할 데이터는 강조점으로 모아 보여주고(Claude만), 해당 차트·표에도 함께 표시됩니다.",
        elem_classes="note",
    )
    ai_provider_radio = gr.Radio(
        choices=["Claude API", "로컬 (Ollama)"], value="Claude API", label="AI 제공자",
    )
    with gr.Group(visible=True) as claude_group:
        ai_model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Claude 모델")
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
            ai_ollama_model_box = gr.Textbox(value=DEFAULT_OLLAMA_MODEL, label="Ollama 모델", scale=2)
            ai_ollama_host_box = gr.Textbox(value=DEFAULT_OLLAMA_HOST, label="Ollama 주소", scale=1)
    ai_insight_btn = gr.Button("AI 인사이트 생성", variant="primary")   # 독립 기능임을 강조하기 위해 primary
    ai_insight_output = gr.Markdown(value="", elem_classes="note")
    ai_insight_state = gr.State(value=None)

ai_provider_radio.change(
    fn=lambda p: (gr.update(visible=(p == "Claude API")), gr.update(visible=(p == "로컬 (Ollama)"))),
    inputs=[ai_provider_radio], outputs=[claude_group, ollama_group],
)
```

**`app.py` — 클릭 핸들러**: 입력 6개(스냅샷·제공자·Claude 모델·API 키·Ollama 모델·Ollama 주소), 출력 5개(해설 텍스트 + 통과수 차트 + 속도 차트 + 존 표 + 트랙 표). 생성 중 버튼을 다시 누르면 같은 요청이 중첩될 수 있으므로, `.click()`을 3단계 체인으로 묶어 생성 중에는 버튼을 비활성화하고 라벨을 바꾼다:

```python
def run_ai_insight(snapshot, provider, claude_model, claude_api_key, ollama_model, ollama_host):
    if not snapshot:
        gr.Warning("먼저 영상을 처리해주세요. 분석 결과가 없습니다.")
        return (gr.skip(),) * 5

    provider_key = "ollama" if provider == "로컬 (Ollama)" else "claude"
    try:
        result = generate_insight(
            snapshot["llm_summary"], provider=provider_key,
            model=claude_model, api_key=(claude_api_key or None),   # 빈 문자열 → None 변환 필수(아래 주의 참고)
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
```
`run_ai_insight()`는 실패 시에도 예외를 던지지 않고 `(gr.skip(),) * 5`를 반환하므로(이전 결과를 그대로 보존), 체인의 마지막 단계(버튼 재활성화)는 성공/실패 모두에서 항상 실행된다.

- **주의 (재현 시 흔히 빠지는 함정)**:
  - `ANTHROPIC_API_KEY`는 `.env` 파일(`.gitignore`에 이미 포함됨)·OS 환경변수, 또는 위 UI 비밀번호 필드로만 설정. **`SessionConfig`(세션 JSON)에는 절대 저장하지 말 것** — 세션 파일을 공유하면 키가 새어나갈 수 있음. UI 입력값은 브라우저 세션 동안만 유지되고 디스크에는 쓰지 않는다.
  - `app.py` 상단에 `from dotenv import load_dotenv; load_dotenv()`를 다른 임포트보다 먼저 실행해야 `ANTHROPIC_API_KEY`가 호출 시점에 채워져 있음.
  - UI의 API 키 입력란은 비워두면 `.env`/환경변수로 자동 폴백하고, 값을 입력하면 그 값이 우선됨 — `api_key=(claude_api_key or None)`로 빈 문자열을 `None`으로 변환해야 폴백이 정상 동작함.
  - `interactive=False`를 빠뜨리면 Gradio가 Styler를 무시하고 일반 값으로 렌더링해 배경색이 안 보일 수 있음 — `zone_df_out`/`track_summary_df_out` 둘 다 확인.
  - `_annotate_chart`/`_highlight_dataframe`는 항상 `ai_insight_state`에 저장된 **원본**(`process_videos()`가 만든 깨끗한 버전)에서 시작해야 함. `run_ai_insight()`가 반환한 이전 결과(이미 주석이 붙은 Figure)를 다시 입력으로 쓰면, 재생성할 때마다 주석이 누적된다.
  - 로컬 모델은 `highlights`가 항상 `[]`이므로 `render_insight_markdown()`이 자동으로 "🔍 주목할 포인트" 블록 없이 `report_markdown`만 반환함 — 별도 분기 불필요.
  - Ollama 모델 태그는 정확히 일치해야 함(`exaone3.5:7.8b`처럼 크기까지 포함). 모델을 `ollama pull` 하지 않은 상태로 호출하면 404가 와서 "pull 실행" 안내 경고가 뜸.
  - `track_summary`는 트랙별 원시 행이 아니라 **클래스별 집계**(`_summarize_track_summary`)만 LLM에 전달됨 — 추적 객체가 수백 개여도 전송량은 클래스 수(보통 2~5개)만큼만 늘어남.
  - pandas Styler의 CSS 문자열은 세미콜론으로 여러 속성을 이어 쓸 수 있음(`"background-color: X; color: Y"`) — 한 속성만 덮어쓰면 나머지는 테마 기본값을 그대로 물려받는다.
  - `thinking`/`output_config.effort`는 의도적으로 사용하지 않음 — 이미 집계된 1~5KB JSON을 요약하는 단발성 작업이라 추론 단계가 품질에 도움이 되지 않고 비용·지연만 늘어남.
- **파일**: `requirements.txt`, `src/analysis/ai_insight.py` (신규), `app.py`

---

### 12. OD 행렬이 항상 비어 있거나 0으로만 나오고, 설정 방법을 화면에서 찾을 수 없음

- **증상**: "OD 행렬" 체크박스를 켜고 처리해도 결과 표가 항상 비어 있거나, 두 존(Zone A/B) 모두에 진입 기록이 있는데도 OD 행렬의 모든 칸이 0으로만 표시됨. 무엇을 더 설정해야 하는지 화면 어디에도 안내가 없었음.
- **원인**: 서로 다른 세 가지 문제가 겹쳐 있었음 — UI/설정 문제 둘과 알고리즘 버그 하나.
  1. **숨겨진 의존성**: `_build_analyzers()`(`src/pipeline.py`)가 `enable_od`와 `zones`뿐 아니라 **`enable_traffic`까지 동시에 켜져 있어야** `ODMatrixBuilder`를 만들었음. OD 행렬은 본질적으로 "존" 기반 기능이라 가상 감지선(교통 분석)과는 무관한데, 같은 아코디언에 체크박스가 있다는 이유만으로 강제 연동되어 있었음.
  2. **발견 불가능한 UI 배치**: "OD 행렬" 체크박스는 "교통 분석" 아코디언에 있었지만, 그 체크박스가 실제로 필요로 하는 "존(zone)" 설정은 완전히 다른 아코디언인 "도시 공간 분석"에 있었음. 두 설정이 서로 다른 섹션에 분리돼 있어 사용자가 둘 다 켜야 한다는 사실을 알 방법이 없었음.
  3. **알고리즘 버그(위 둘을 고친 뒤에도 결과가 0으로만 나옴)**: `ODMatrixBuilder.update()`(`src/analysis/od_matrix.py`)가 트랙의 "이전 존"을 **바로 직전 프레임의 존**으로만 기억하고 있었음. 두 존 사이에 존이 아닌 일반 도로 구간이 있으면(현실에서는 거의 항상 그러함), 그 구간을 지나는 프레임마다 `_track_in_zone[tid] = None`으로 덮어써져 "Zone A에 있었다"는 기억이 사라짐. 그 결과 다음 존(Zone B)에 진입한 순간엔 `prev`가 이미 `None`이라 전이가 절대 기록되지 않음 — 두 존이 서로 붙어 있어 한 프레임 안에 바로 옮겨가는 경우에만 정상 작동하는 구조였음.
- **해결**:

**1) `enable_traffic` 의존성 제거** — OD 행렬은 `존` + `enable_od`만으로 독립 동작:

```python
# src/pipeline.py — _build_analyzers()
zones = analysis_config.get('zones', [])
if zones and analysis_config.get('enable_od'):      # enable_traffic 조건 삭제
    from .analysis.od_matrix import ODMatrixBuilder
    analyzers.append(ODMatrixBuilder(zones))
```

**2) 체크박스를 실제 의존 대상 옆으로 이동** — "OD 행렬 계산" 체크박스를 "교통 분석"에서 빼내 "도시 공간 분석"의 존 정의 바로 아래로 옮기고, "최소 2개 이상의 존 필요"·"도시 분석 활성화와는 무관하게 독립 동작" 안내문을 함께 추가. (이동만 하면 됨 — `process_videos()`의 매개변수 순서나 `inputs=[...]` 리스트는 그대로 둬도 됨. Gradio 와이어링은 컴포넌트가 어느 `with gr.Accordion(...)` 블록에 있는지가 아니라 Python 변수 자체를 참조하기 때문.)

**3) 빈 결과에 대한 안내** — 결과가 비어 있는 상황을 더 이상 빈 표로 침묵 처리하지 않고, 원인을 구분해 안내 메시지를 표시:

```python
# app.py — process_videos()
od_df = None
od_data = last_analysis.get('od_matrix', {})
if od_data and 'matrix_df' in od_data:
    od_df = od_data['matrix_df'].reset_index()
    od_df.rename(columns={'index': 'Origin \\ Dest'}, inplace=True)
elif enable_od:
    if len(zones) < 2:
        od_msg = "OD 행렬에는 최소 2개 이상의 존이 필요합니다. '도시 공간 분석' 아코디언에서 존을 2개 이상 활성화하고 좌표를 입력해주세요."
    else:
        od_msg = "이번 영상에서는 활성화된 존 사이를 이동한 객체가 감지되지 않았습니다."
    od_df = pd.DataFrame({"안내": [od_msg]})
```

**4) 알고리즘 버그 수정** — 트랙이 어떤 존에도 속하지 않는 프레임은 그냥 건너뛰도록 변경. `_track_in_zone[tid]`는 트랙이 **마지막으로 확인된 존**만 기억하고, 존이 아닌 구간을 지나도 지워지지 않음:

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
  - 위 4가지를 **모두** 적용해야 함 — 1·2만 고치면 체크박스를 찾고 켤 수는 있지만 알고리즘 버그 때문에 결과가 계속 0으로 나오고, 4만 고치면 애초에 체크박스/존 설정을 못 찾거나 `enable_traffic`이 꺼져 있으면 분석기 자체가 안 만들어져서 여전히 아무 결과도 안 나옴.
  - 존(zone)은 "도시 분석 활성화"(`enable_urban`) 체크박스와 무관하게 항상 정의 가능 — `enable_urban`은 `ZoneAnalyzer`/히트맵에만 영향을 주고, OD 행렬은 별도로 `zones` 리스트 자체만 본다.
  - `origin`은 트랙이 **최초로** 진입한 존으로 고정되고 이후 다른 존에 들어가도 갱신되지 않음(의도된 동작) — 즉 A→B→C로 이동하면 (A,B)와 (A,C) 둘 다 기록되고 (B,C)는 기록되지 않음. "최초 출발지 기준" OD 행렬이라는 설계이며, 알고리즘 수정은 이 설계 자체가 아니라 "존 사이 공백 구간에서 기억이 끊기는" 버그만 고친 것.
  - 같은 존 안에 머무는 동안은(`prev == zone`) 매 프레임 중복 카운트되지 않음.
  - OD 행렬이 0이 아닌 값을 가지려면 같은 트랙이 서로 다른 두 존을 실제로 통과해야 함 — 존 1개만 활성화하면 모든 진입이 "같은 곳"이라 전이가 기록되지 않아 항상 빈 결과가 나옴(이건 버그가 아니라 정상 동작).
- **파일**: `src/pipeline.py`, `src/analysis/od_matrix.py`, `app.py`

---

### 13. 배치 처리 시 마지막 영상의 결과만 표시되고, 다른 영상은 확인할 수 없음

- **증상**: 여러 영상을 배치로 처리하면 "영상별 처리 요약" 표와 ZIP 다운로드는 모든 영상을 포함하지만, 위쪽 결과 영역(영상·차트·표·AI 인사이트)에는 **마지막으로 처리된 영상의 결과만** 표시됨. 다른 영상의 결과를 보려면 ZIP을 받아 직접 열어보는 것 외에는 방법이 없었음.
- **원인**: `process_videos()`가 영상 루프(`all_results`)를 다 돈 뒤 `last_vpath, last_result = all_results[-1]`로 **마지막 항목만** 꺼내 차트(`_make_traffic_count_chart` 등)·표(OD/존/트랙 요약)·AI 인사이트 요약을 빌드했음. 나머지 영상들의 분석 결과는 ZIP에 파일로만 묻히고, 화면에 다시 불러올 방법이 전혀 없었음.
- **해결**: 영상별로 화면 표시용 산출물(차트·표·AI 인사이트 스냅샷)을 묶는 `_build_video_package()` 함수를 새로 만들어, 루프가 끝난 뒤 **모든 영상에 대해** 호출하고 그 결과 리스트(`packages`)를 새 `gr.State`(`batch_packages_state`)에 저장. 기본 화면은 여전히 마지막 영상(`packages[-1]`)을 보여주되, 드롭다운 + 버튼으로 다른 영상으로 전환 가능. 이 선택 UI는 **"추적 결과" 영상/궤적 이미지 바로 아래, 모든 분석 결과 아코디언보다 위**에 둔다 — 분석 결과를 한참 내려서 보기 전에 먼저 어떤 영상을 볼지 고르는 흐름이 자연스럽고, 선택을 바꿀 때마다 아래로 스크롤할 필요가 없기 때문이다.

**신규 함수 — `_build_video_package()`**: 기존에 `process_videos()` 안에서 "마지막 영상"에 대해 한 번만 하던 차트/표/AI 스냅샷 빌드 로직(항목 12의 OD 빈 결과 안내 포함)을 함수로 추출한 것 — `_make_traffic_count_chart`/`_make_speed_chart`(항목 10)/`build_analysis_summary`(항목 11)는 모두 그대로 재사용:

```python
# app.py — process_videos() 정의보다 앞에 추가
def _build_video_package(vpath, result, zones, enable_od, enabled_flags, label):
    """단일 영상의 처리 결과로부터 화면에 표시할 차트·표·AI 인사이트 스냅샷을 만든다."""
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
            od_msg = ("OD 행렬에는 최소 2개 이상의 존이 필요합니다. "
                      "'도시 공간 분석' 아코디언에서 존을 2개 이상 활성화하고 좌표를 입력해주세요.")
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
                '존': zs['zone'], '입장 횟수': zs['entry_count'],
                '평균 체류 (s)': zs['mean_dwell_s'], '최대 체류 (s)': zs['max_dwell_s'],
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
        "video": result['video'], "csv": result['csv'], "trajectory_img": result['trajectory_img'],
        "heatmap_img": result.get('heatmap_img'), "excel": result.get('excel'), "charts": result.get('charts'),
        "traffic_count_fig": traffic_count_fig, "traffic_speed_fig": traffic_speed_fig,
        "od_df": od_df, "zone_df": zone_df, "track_summary_df": track_summary_df,
        "ai_insight_snapshot": ai_insight_snapshot,
    }
```

**`process_videos()`에서 모든 영상에 대해 호출**:

```python
# app.py — process_videos() 안, 영상 처리 루프(all_results) 다음
enabled_flags = {
    "enable_traffic": bool(enable_traffic),
    "enable_speed": bool(enable_speed) and scale_mpp is not None,
    "enable_od": bool(enable_od),
    "enable_urban": bool(enable_urban),
}
packages = [
    _build_video_package(vp, res, zones, bool(enable_od), enabled_flags,
                          label=f"{idx+1:02d}. {Path(vp).name}")
    for idx, (vp, res) in enumerate(all_results)
]
last_pkg = packages[-1]
...
batch_video_choices = gr.update(choices=[pkg['label'] for pkg in packages], value=last_pkg['label'])
# 18 outputs: 기존 16개 + packages(batch_packages_state) + batch_video_choices(batch_video_select)
# 최종 return의 각 자리는 last_pkg['video'], last_pkg['csv'], ..., last_pkg['ai_insight_snapshot'], packages, batch_video_choices
```

```python
# app.py — UI ("추적 결과" 헤더 아래, video_output/summary_img Row 바로 다음,
# "교통 분석 결과" 등 분석 아코디언들보다 앞)
gr.Markdown(
    "**배치 처리 결과 선택** — 여러 영상을 배치로 처리한 경우, 아래에서 영상을 골라 "
    "이 페이지 전체(차트·표·AI 인사이트)를 그 영상 기준으로 바꿔 볼 수 있습니다. "
    "기본값은 마지막으로 처리된 영상입니다.",
    elem_classes="note",
)
with gr.Row():
    batch_video_select = gr.Dropdown(label="결과를 확인할 영상 선택", choices=[], value=None, scale=3)
    batch_view_btn = gr.Button("선택한 영상 결과 보기", scale=1, variant="primary")
batch_packages_state = gr.State(value=None)

# "배치 처리 결과" 섹션(맨 아래, 영상별 처리 요약 표 + ZIP 다운로드)에는
# 선택 UI를 두지 않고, 위쪽으로 이동했다는 안내만 남긴다.

def view_batch_video(packages, label):
    if not packages or not label:
        gr.Warning("선택할 영상이 없습니다. 먼저 배치 처리를 완료해주세요.")
        return (gr.skip(),) * 14
    pkg = next((p for p in packages if p["label"] == label), None)
    if pkg is None:
        gr.Warning("선택한 영상의 결과를 찾을 수 없습니다.")
        return (gr.skip(),) * 14
    return (pkg["video"], pkg["csv"], pkg["trajectory_img"], f"{label}  결과 표시 중",
            pkg["traffic_count_fig"], pkg["traffic_speed_fig"], pkg["od_df"],
            pkg["heatmap_img"], pkg["zone_df"], pkg["track_summary_df"],
            pkg["excel"], pkg["charts"], pkg["ai_insight_snapshot"], "")

batch_view_btn.click(
    fn=view_batch_video,
    inputs=[batch_packages_state, batch_video_select],
    outputs=[video_output, csv_download, summary_img, status_box,
             traffic_count_plot, traffic_speed_plot, od_df_out,
             heatmap_img_out, zone_df_out, track_summary_df_out,
             excel_download, charts_download, ai_insight_state, ai_insight_output],
)
```

- **부가 효과**: `ai_insight_state`도 영상을 전환할 때 함께 바뀌므로, 드롭다운으로 영상을 고른 뒤 "AI 인사이트 생성"을 누르면 **그 영상**에 대한 해설이 새로 생성됨(이전 영상의 해설 텍스트는 전환 시 자동으로 비움).
- **주의 (재현 시 흔히 빠지는 함정)**:
  - `process_videos()`의 출력 개수가 16개 → 18개로 늘어남 — `_all_outputs`, `video_input.clear()`의 초기화 튜플, "영상 업로드 안 함" 에러 반환 튜플까지 **세 군데 모두** 같은 길이로 맞춰야 함. 길이가 안 맞으면 Gradio가 와이어링 단계에서 조용히 잘못된 컴포넌트에 값을 매핑하거나 에러를 던짐.
  - 드롭다운 라벨은 `"01. 파일명.mp4"`처럼 순번을 앞에 붙임 — 배치 파일 중 이름이 같은 영상이 섞여 있어도 라벨이 겹치지 않도록 하기 위함.
  - 단일 영상 처리 시에는 `is_batch`가 False라 드롭다운 choices가 빈 채로 남음(기존 `batch_summary_df_out`/`batch_zip_download`가 단일 처리 시 비어 있는 것과 동일한 동작) — 버그가 아니라 배치 전용 기능이라는 의미.
  - 선택 UI(드롭다운+버튼+`batch_packages_state`)를 어디에 배치하든 `view_batch_video()`의 동작이나 `.click()` 와이어링에는 영향이 없음 — Gradio는 컴포넌트가 Python 변수로 참조되는지만 보고, 어느 `with gr.Row()`/`with gr.Column()` 블록 안에 있는지는 무관하다(항목 11의 아코디언 분리 때와 같은 원리).
- **파일**: `app.py`

---

### 14. 궤적 요약 이미지가 객체 레이블 범례 때문에 세로로 과도하게 늘어남

- **증상**: `trajectory_summary.png`에서 추적 객체 수가 많을 때(수십 개) 이미지 우측에 트랙ID별 범례(`#51`, `#52`, ...)가 한 줄씩 표시되는데, matplotlib이 `bbox_inches='tight'`로 저장하면서 그 범례를 전부 담으려고 이미지 높이를 범례 길이만큼 늘려버림 — 실제 궤적 그림은 위쪽 일부일 뿐인데 그 아래로 새까만 빈 공간이 수천 픽셀 이어짐.
- **원인**: `SummaryImageExporter.save()`(`src/visualization/exporter.py`)가 트랙마다 `mpatches.Patch(color=c, label=f'#{tid}')`를 만들어 `ax.legend(...)`로 전부 표시하고 있었음. 트랙 수에 비례해 범례 칸 수가 늘어나는데 폭은 그대로라 한 칸씩 아래로 쌓이며 이미지 전체가 세로로 늘어짐.
- **해결**: 범례(객체별 레이블) 자체를 제거. 각 트랙은 여전히 고유 색상으로 구분되어 경로/예측선이 그려지지만, 어떤 색이 어떤 트랙 ID인지 알려주는 텍스트 목록은 더 이상 그리지 않음:

```python
# src/visualization/exporter.py — SummaryImageExporter.save()
# (mpatches.Patch 수집 + ax.legend(...) 호출을 완전히 삭제하고, 선/마커만 그린다)
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
```
파일 맨 위 `import matplotlib.patches as mpatches`도 더 이상 쓰이지 않으므로 함께 제거.

검증: 트랙 69개를 시뮬레이션해 저장한 결과 이미지 크기가 `1785×1183`으로 정상 범위에 머무는 것을 확인(범례가 있을 때는 트랙 수에 비례해 수천 픽셀까지 늘어났음).

- **주의 (재현 시 흔히 빠지는 함정)**:
  - 범례를 지워도 트랙 구분 자체는 그대로 유지됨 — `self.colors.get(tid, ...)`로 트랙마다 고유 색을 쓰는 로직은 손대지 않았으므로, "어떤 트랙인지"는 색으로 구분 가능하고 "이 색이 트랙 몇 번인지" 텍스트만 빠진 것.
  - 이 이미지는 `process_videos()`의 결과 화면(`summary_img`)뿐 아니라 배치 ZIP에도 `trajectory.png`로 포함되므로, 두 경로 모두에서 동일하게 정상 크기로 나오는지 확인할 것.
- **파일**: `src/visualization/exporter.py`

---

### 15. 배치 처리 시 세션 파일이 생성되지 않음

- **증상**: 배치 처리(여러 영상 동시 업로드) 후에는 "세션 파일 (JSON)" 다운로드가 항상 비어 있음(단일 영상 처리 시에는 정상 생성됨).
- **원인**: `process_videos()`의 세션 저장 블록이 `if not is_batch:`로 감싸여 있어, 배치 처리 시에는 의도적으로 건너뛰고 있었음(`session_out = None`).
- **해결**: 세션 저장을 배치/단일 공통 로직으로 변경. 배치는 영상이 여러 개라 `video_hash`(특정 영상 하나를 가리키는 해시) 필드는 의미가 없으므로 비워두고, 대신 처리된 파일 목록을 `notes`에 덧붙이고 `stats`에는 전체 영상의 합산 프레임·트랙 수를 기록:

```python
# app.py — process_videos()
# total_frames/total_tracks는 is_batch 분기 밖에서 항상 계산해둔 값(단일 영상이면 영상 1개분의 합과 같음)
notes_text = session_notes or ''
if is_batch:
    file_list = ', '.join(Path(vp).name for vp, _ in all_results)
    batch_tag = f"[배치 처리 — {total_videos}개 영상: {file_list}]"
    notes_text = f"{notes_text}\n{batch_tag}" if notes_text else batch_tag

tmp_dir = Path(tempfile.mkdtemp())
session_path = tmp_dir / "session.json"
session_cfg = SessionConfig(
    video_hash=('' if is_batch else AnalysisSession.compute_video_hash(last_vpath)),
    yolo_model=yolo_model, conf_thresh=conf_thresh, seq_len=int(seq_len), pred_len=int(pred_len),
    lstm_mode=mode, class_filter=list(class_filter) if class_filter else [],
    counting_lines=counting_lines, zones=zones, zone_areas=zone_areas, scale_mpp=scale_mpp,
    enable_traffic=bool(enable_traffic), enable_speed=bool(enable_speed), enable_od=bool(enable_od),
    enable_urban=bool(enable_urban), enable_heatmap=bool(enable_heatmap),
    heatmap_classes=list(heatmap_classes) if heatmap_classes else [],
    export_format=export_format, enable_mc_dropout=bool(enable_mc_dropout), chart_format=chart_format,
    notes=notes_text,
    stats={'total_frames': total_frames, 'total_tracks': total_tracks, 'device': DEVICE_DESC},
)
AnalysisSession.save(session_path, session_cfg)
session_out = str(session_path)   # is_batch 분기 없이 항상 실행
```
- **주의 (재현 시 흔히 빠지는 함정)**:
  - `restore_session()`은 `video_hash`를 현재 업로드된 영상과 비교·검증하지 않고 단순히 메타데이터로만 저장함 — 배치 세션에서 비워둬도 설정 복원에는 전혀 영향 없음.
  - `total_frames`/`total_tracks`는 기존에 `is_batch` 분기 안에서만 계산되던 것을 분기 밖으로 빼서 항상 계산하도록 바꿔야 함 — 단일 영상 처리 시의 상태 텍스트(`status`)는 여전히 `last_stats`를 쓰므로 그쪽 동작에는 변화가 없음.
  - 출력 개수나 `_all_outputs` 목록에는 변화가 없음 — `session_download` 슬롯 자체는 항목 10 이전부터 이미 존재했고, 이번 수정은 그 슬롯에 값을 채우는 조건만 바꾼 것.
- **파일**: `app.py`

---

### 16. AI 인사이트가 1분짜리 영상을 "61분 1초"라고 잘못 보고함

- **증상**: 실제로는 약 1분(1833프레임, 30fps → 61.1초) 분량인 영상을 처리했는데, AI 인사이트 리포트에 "총 61분 1초 분량의 영상..."이라고 적힘 — 실제 길이의 61배.
- **원인**: 데이터 자체는 정상이었음. Gradio 업로드 캐시에 남아있던 실제 영상 파일을 직접 `cv2`로 열어 확인한 결과 `fps=30, frame_count=1833` → `duration_s = 1833 / 30 = 61.1`(초)로 정확히 계산되어 있었음(1분 1.1초). 문제는 `build_analysis_summary()`(`src/analysis/ai_insight.py`)가 이 값을 `"duration_s": 61.1`(초 단위 원시 숫자)로만 Claude에 전달하고, 사람이 읽기 좋은 "N분 M초" 형태로 바꾸는 일을 Claude에게 맡겼다는 것 — Claude가 61.1을 60으로 나누는 환산(÷60)을 틀려서 "61분 1초"라고 적어버림. LLM에게 모듈러(÷60) 연산을 텍스트 생성 중에 시키면 이런 산술 오류가 종종 발생한다.
- **검증**: 의심되는 원본 영상을 Gradio 임시 업로드 캐시(`%TEMP%\gradio\...`)에서 찾아 `cv2.VideoCapture`로 직접 열어 `fps`/`frame_count`를 확인 → `duration_s` 계산 자체가 옳다는 것을 먼저 확인한 뒤, 어디서 "61분"이 나왔는지를 역추적해 원인을 LLM 쪽 환산 오류로 좁힘.
- **해결**: 분/초 환산을 LLM에게 시키지 않고 Python에서 미리 끝내, 이미 포맷된 문자열을 그대로 쓰라고 명시:

```python
# src/analysis/ai_insight.py
def _format_duration(duration_s: float) -> str:
    """초 단위 길이를 "N분 M초"(1분 미만이면 "M초") 형태의 문자열로 변환한다."""
    total_seconds = int(round(duration_s or 0))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}분 {seconds}초" if minutes else f"{seconds}초"


def build_analysis_summary(last_analysis, last_result, enabled_flags) -> dict:
    ...
    duration_s = round(last_result.get("duration_s", 0) or 0, 1)
    summary = {
        "meta": {
            "total_frames": stats.get("total_frames"),
            "total_tracks": stats.get("total_tracks"),
            "duration_s": duration_s,
            "duration_formatted": _format_duration(duration_s),   # ← 신규
            "enabled_analyses": {k: bool(v) for k, v in (enabled_flags or {}).items()},
        },
    }
```

프롬프트 지침(`_prompt_preamble()`의 `notes`)에 한 줄 추가:

```python
"- 영상 길이를 언급할 때는 meta.duration_formatted 값을 그대로 가져와 쓰세요. "
"meta.duration_s(초 단위 원시값)를 분·초로 직접 환산하지 마세요 — 환산 과정에서 계산 오류가 날 수 있습니다.\n"
```

검증: `_format_duration(61.1)` → `"1분 1초"`, `_format_duration(3661.0)` → `"61분 1초"`(실제로 61분짜리 영상이면 이렇게 나오는 게 맞음), `_format_duration(119.6)` → `"2분 0초"`로 모두 정확히 환산됨을 확인.

- **주의 (재현 시 흔히 빠지는 함정)**:
  - `duration_s`는 그대로 남겨둠(다른 코드가 초 단위 정밀도를 원할 수 있으므로) — 다만 프롬프트에는 "이 값으로 분·초를 직접 계산하지 말 것"이라는 지침이 반드시 함께 있어야 함. 필드만 추가하고 지침을 안 넣으면 모델이 여전히 `duration_s`를 보고 직접 환산을 시도할 수 있음.
  - 이런 부류의 버그(LLM이 받은 숫자를 변환하다가 산술을 틀리는 것)는 데이터 파이프라인을 아무리 들여다봐도 안 보임 — 실제 입력 영상의 메타데이터를 직접 확인해 "우리가 보낸 숫자가 맞다"는 것부터 확인하고, 그 다음 "LLM이 그 숫자를 어떻게 잘못 다뤘는지"로 좁혀가는 순서가 중요함.
  - `_format_duration`은 `round()`로 반올림한 정수 초를 기준으로 분·초를 나누므로, 0.5초 미만의 오차는 무시됨(리포트용 표시이므로 문제 없음).
- **파일**: `src/analysis/ai_insight.py`

---

### 17. "궤적 요약" 이미지에 GPU 사용률 차트가 표시됨

- **증상**: 처리 완료 후 "궤적 요약" 이미지 패널에 트랙 경로 그림이 아니라 "GPU 사용률" 모니터링 차트(주황색 선 그래프)가 표시됨. 화면 자체에는 정상적인 라벨("궤적 요약")이 붙어 있어 단순 레이블 오타처럼 보이지만, 실제로는 그 라벨이 가리키는 PNG 파일의 **내용**이 통째로 바뀌어 있던 것이었음.
- **원인**: `SummaryImageExporter.save()`(`src/visualization/exporter.py`)가 `plt.tight_layout()`/`plt.savefig(path, ...)`/`plt.close()`를 **인자 없이** 호출하고 있었음 — 이 pyplot 레벨 함수들은 자신이 만든 `fig` 객체가 아니라 matplotlib의 **전역 "현재 figure"**(`plt.gcf()`)에 대해 동작함. 그런데 `app.py`의 실시간 시스템 모니터링(`_update_monitor()` → `_make_chart()`, CPU·RAM·GPU 차트)은 `gr.Timer(value=2.0, active=True)`로 **영상 처리 중에도 2초마다 계속 실행**되며 자체적으로 `plt.subplots()`를 호출함. `process_videos()`(영상 처리, 길 때는 수 분 소요)는 별도 워커 스레드에서 동작하는 동안 모니터링 타이머는 같은 프로세스의 다른 스레드에서 계속 돌고 있어, **두 스레드가 matplotlib의 같은 전역 pyplot 상태를 동시에 건드림**. 트랙 요약 이미지를 만드는 도중(`plt.subplots()` 호출 후, `plt.savefig()` 호출 전 사이) 모니터링 타이머의 `plt.subplots()`가 끼어들면 "현재 figure"가 GPU 차트로 바뀌어버리고, 그 뒤에 호출된 `plt.savefig(path)`가 트랙 요약이 아니라 GPU 차트를 그 경로에 저장해버림. matplotlib의 pyplot 전역 상태는 스레드 안전하지 않다.
- **해결**: 전역 상태에 의존하지 않도록, `plt.subplots()`가 반환한 **그 `fig` 객체에 직접** 호출하도록 변경:

```python
# src/visualization/exporter.py — SummaryImageExporter.save()
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
...
fig.tight_layout()                                                       # plt.tight_layout() 대신
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')     # plt.savefig(path, ...) 대신
plt.close(fig)                                                           # plt.close() 대신 — 닫을 figure를 명시
```
`app.py`의 `_make_chart()`는 이미 `fig.tight_layout(pad=0.4)`처럼 객체에 직접 호출하고 있어 문제가 없었음(전역 상태를 더럽히는 쪽은 아님 — 다만 같은 전역 레지스트리에 새 figure를 계속 추가하긴 함).

검증: 영상 처리 스레드를 흉내 내어 `SummaryImageExporter.save()`를 반복 호출하는 동안, 다른 스레드에서 `plt.subplots()`를 1ms 간격으로 계속 호출해 경쟁 상태를 강제로 재현 — 수정 전 코드 경로에서는 다른 figure가 저장될 위험이 있고, 수정 후에는 50회 반복 모두 정상적으로 궤적 그림만 저장됨을 확인.

- **주의 (재현 시 흔히 빠지는 함정)**:
  - matplotlib을 여러 스레드에서 동시에 쓸 때는 **인자 없는 pyplot 레벨 함수**(`plt.savefig()`, `plt.close()`, `plt.tight_layout()`, `plt.gcf()`, `plt.gca()` 등)를 절대 쓰지 말고, 항상 `fig`/`ax` 객체에 직접 호출(`fig.savefig()`, `fig.tight_layout()`, `plt.close(fig)`)할 것 — 이건 이 프로젝트뿐 아니라 matplotlib을 쓰는 모든 멀티스레드 코드에 적용되는 일반 원칙.
  - 이 버그는 **항상** 재현되지 않음(타이밍에 의존하는 경쟁 상태) — 모니터링 타이머가 하필 그 짧은 순간에 끼어들 때만 발생하므로, 같은 영상을 다시 처리하면 정상적으로 나올 수도 있음. "가끔 발생하고 재현이 불안정한 시각적 버그"는 출력 와이어링(라벨/순서) 문제보다 이런 동시성 문제부터 의심해볼 것.
  - 같은 문제가 다른 곳에도 있을 수 있으므로 `import matplotlib`/`from matplotlib`가 있는 모든 파일(`grep -rln "import matplotlib" src/ app.py`)을 확인했음 — 현재는 `app.py`(`_make_chart`, 이미 안전)와 `src/visualization/exporter.py`(이번에 수정) 두 곳뿐.
- **파일**: `src/visualization/exporter.py`

### 18. 속도 분포(박스플롯) 차트가 이상치로 뒤덮여 깨진 것처럼 보임

- **증상**: `research_charts.html`의 "Speed Distribution by Class" 박스플롯에서 클래스별 박스가 0~8km/h 부근에 짜부러져 있고, 그 위로 점들이 빽빽하게 겹쳐 찍혀 막대/기둥처럼 보임. 정상적인 주행속도(15~60km/h)가 전부 박스 위쪽 "이상치"로 그려져 분포가 깨진 것처럼 보였음.
- **원인**: `PlotlyExporter._speed_box()`(`src/visualization/stats_exporter.py`)가 트랙(차량)별 대표 속도가 아니라, 트랙이 살아있는 동안의 **모든 프레임의 순간속도**(EMA smoothed)를 그대로 풀링해서 박스플롯에 넘기고 있었음. 차량 한 대가 정차·서행하는 프레임 수는 수백 개인 반면 실제로 빠르게 주행하는 프레임 수는 적기 때문에, 프레임 단위로 합치면 "체류 시간"이 분포 모양을 지배하게 됨. 실제 데이터에서 car 클래스는 프레임 샘플 16,185개 중 1,469개(9.1%)가 IQR 기준 이상치로 분류되어 박스 위에 점이 두껍게 쌓였고, 평균(6.17km/h)보다 표준편차(7.58km/h)가 더 큰 극단적 우측 편포를 보였음. `exporter.py`의 Track Summary 시트는 이미 트랙당 `mean_speed_kmh` 하나를 집계해 쓰고 있어, 이 차트만 다른 집계 단위(프레임)를 쓰고 있었던 셈.
- **검증**: HTML에 내장된 Plotly JSON에서 실제 트레이스 데이터를 추출해 클래스별 표본 수·분위수·이상치 비율을 계산(car 16,185개/9.1% 이상치, truck 2,132개/7.0% 등). 합성 데이터(트랙당 정차 프레임 다수 + 주행 프레임 소수)로 수정 후 `_speed_box()`를 호출해, y 배열 길이가 프레임 수가 아니라 트랙 수와 같아짐(트랙 50개 → 표본 50개)을 확인.
- **해결**: 프레임을 그대로 풀링하지 않고, 트랙별 평균 속도 1개를 그 트랙의 대표값으로 쓴 뒤 클래스별로 모음:

```python
# src/visualization/stats_exporter.py — PlotlyExporter._speed_box()
by_class = {}
for tid, speeds in track_speeds.items():
    if not speeds:
        continue
    cls = track_cls.get(tid, 'unknown')
    by_class.setdefault(cls, []).append(float(np.mean(speeds)))  # 프레임 전체 대신 트랙당 평균 1개
...
fig.update_layout(title='Speed Distribution by Class (km/h)',
                  yaxis_title='Mean Speed per Vehicle (km/h)', template='plotly_white')
```
- **주의 (재현 시 흔히 빠지는 함정)**:
  - 같은 "프레임 전체 풀링" 패턴이 `src/analysis/speed_estimator.py`의 `finalize()`(Excel "Speed Statistics" 시트의 mean/std/percentile)에도 동일하게 존재했음 — [19]에서 같은 방식으로 수정함.
  - 박스플롯의 "이상치"는 통계적으로 드문 사건이 아니라, 샘플링 단위(프레임 vs 차량)를 잘못 잡았을 때도 대량으로 발생할 수 있음 — 이상치 비율이 비정상적으로 높다면(수% 이상) 우선 "표본 하나가 무엇을 대표하는가"부터 의심할 것.
- **파일**: `src/visualization/stats_exporter.py`

### 19. 엑셀 "Speed Statistics" 시트도 같은 프레임 풀링 왜곡을 갖고 있었음

- **증상**: 18번과 별개로, Excel 내보내기의 "Speed Statistics" 시트(`per_class`의 mean/std/p15/p50/p85/p95/max)도 표준편차가 평균보다 크게 나오는 등 비정상적으로 편향된 값을 보임. P85는 도로교통 분야 표준 지표(85th percentile speed, 시트 작성 시 강조 표시되는 컬럼)인데, 실제로는 "차량별" 속도가 아니라 "프레임별" 순간속도의 85번째 백분위수를 계산하고 있어 의미가 달라짐.
- **원인**: `SpeedEstimator.finalize()`(`src/analysis/speed_estimator.py`)가 18번과 동일한 패턴 — 트랙별 모든 프레임 속도를 `by_class[cls].extend(speeds)`로 그대로 풀링한 뒤 mean/std/percentile을 계산함. `count` 필드도 사실은 "프레임 샘플 수"였지 "차량 수"가 아니었음(예: car 16,185 = 프레임 수, 실제 차량 수는 그보다 훨씬 적음). `ai_insight.py`의 `_summarize_speed()`가 이 `per_class`를 그대로 LLM에 넘기므로, AI 인사이트 문장도 왜곡된 통계를 근거로 생성될 수 있었음. 같은 파일의 `_summarize_track_summary()`는 이미 트랙당 `mean_speed_kmh` 하나를 쓰고 있어 일관성이 없었음.
- **검증**: 합성 데이터(트랙당 정차 프레임 다수 + 주행 프레임 소수, 트랙 50개)로 수정 후 `finalize()`를 호출 — `count=50`(트랙 수)과 일치하고, std(1.9)가 mean(5.6)보다 작아져 정상적인 분포로 돌아옴을 확인.
- **해결**: 18번과 동일하게, 트랙별 평균 속도 1개를 대표값으로 모아서 통계를 계산하도록 변경:

```python
# src/analysis/speed_estimator.py — SpeedEstimator.finalize()
by_class = defaultdict(list)
for tid, speeds in self.track_speeds.items():
    if not speeds:
        continue
    cls = self.track_cls.get(tid, 'unknown')
    by_class[cls].append(float(np.mean(speeds)))  # 프레임 전체 대신 트랙당 평균 1개
```
- **주의 (재현 시 흔히 빠지는 함정)**:
  - 이 변경으로 `per_class[cls]['count']`의 의미가 "프레임 샘플 수"에서 "해당 클래스 차량 수"로 바뀜 — 더 올바른 의미지만, 이전에 생성된 엑셀 결과와 숫자를 직접 비교하면 줄어든 것처럼 보일 수 있으니 영상을 재처리한 뒤 비교할 것.
  - `track_speeds`/`track_cls`(프레임 단위 원시 데이터)는 반환값에 그대로 남아있음 — 새로 집계 코드를 추가할 때 또 같은 실수(프레임 그대로 풀링)를 반복하지 않도록 주의.
- **파일**: `src/analysis/speed_estimator.py`

### 20. 단일 영상 업로드 시 "video not playable" 오류가 간헐적으로 발생(배치 탭은 정상)

- **증상**: "단일 영상" 탭에 정상적인(손상되지 않은) 영상 파일을 업로드해도 화면에 빨간 오류 메시지로 "video not playable"이 간헐적으로 표시됨. 같은 영상을 "배치 처리" 탭(`gr.File`)에 올리면 오류 없이 정상 동작함.
- **원인**: "단일 영상" 탭이 `gr.Video` 컴포넌트(`app.py:784`)를 쓰고 있었는데, 이건 업로드된 파일을 브라우저의 HTML5 `<video>` 태그로 실제 디코딩·재생까지 시도하는 컴포넌트임. OpenCV/YOLO 백엔드 처리는 거의 모든 코덱을 문제없이 읽지만, 일부 영상(예: OpenCV `mp4v`로 인코딩된 mp4, 특정 카메라/저장 장치의 코덱)은 Chrome/Edge의 `<video>` 태그가 지원하지 않는 비디오 코덱을 쓰고 있어서 **프리뷰 재생만** 실패함. 이 실패가 "video not playable" 토스트로 표시되며 업로드 자체가 막힘. 영상마다 코덱이 다르므로 "정상 파일인데도 간헐적"으로 보였던 것. 이 프로젝트는 ffmpeg가 시스템 PATH에 설치되어 있지 않아(`imageio-ffmpeg`로 번들된 바이너리만 있음, `src/visualization/exporter.py`의 `_get_ffmpeg_exe()` 참고), Gradio가 내부적으로 코덱을 웹 호환 포맷으로 재인코딩해서 우회하는 것도 기대할 수 없는 환경이었음. 배치 탭은 `gr.File`을 쓰기 때문에 브라우저가 재생을 시도하지 않아 이 문제 자체가 발생하지 않았음.
- **검증**: `gr.Video.preprocess()`/`gr.File.preprocess()`를 합성 mp4 파일로 직접 호출해 두 컴포넌트의 동작 차이를 확인 — `gr.Video`는 (이 환경에서는 `include_audio` 기본값이 `True`로 해석되어 ffmpeg 재인코딩 자체는 발생하지 않지만) 프런트엔드의 `<video>` 디코딩에 의존하는 구조이고, `gr.File`은 파일을 그대로 경로로 전달해 디코딩을 전혀 시도하지 않음을 확인.
- **해결**: 처리 백엔드는 영상을 브라우저에서 미리 재생할 필요가 없으므로, "단일 영상" 탭도 배치 탭과 동일하게 `gr.File`로 교체:

```python
# app.py — "단일 영상" 탭
video_input = gr.File(
    file_types=[".mp4", ".avi", ".mov", ".mkv"],
    label="영상 파일  (MP4 / AVI / MOV)",
)
```
경로 추출 코드도 `gr.Video`가 과거 반환하던 dict 형태를 풀어주던 방어 코드(`vp.get("video", {})...`)가 더 이상 의미가 없어져 `paths = [str(video_input)]`로 단순화함(`gr.File`은 경로 문자열을 그대로 반환).
- **주의 (재현 시 흔히 빠지는 함정)**:
  - 이 변경으로 "단일 영상" 탭에서 업로드한 영상의 미리보기 플레이어가 사라짐(파일명만 표시) — 이 앱은 감지선/존 좌표를 영상 위 클릭이 아니라 별도 숫자 입력으로 받기 때문에 기능상 손실은 없음.
  - "video not playable"은 Gradio/브라우저의 **프리뷰 전용** 오류이며 파일 자체의 손상이나 백엔드 처리 가능 여부와는 무관함 — 영상 관련 "간헐적" 오류를 볼 때 파일 손상을 의심하기보다, 먼저 어떤 컴포넌트(`gr.Video` vs `gr.File`)로 입력받는지부터 확인할 것.
- **파일**: `app.py`

---

## 라이선스

MIT
