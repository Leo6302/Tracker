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

## 라이선스

MIT
