# Project State

_Reconstructed from HANDOFF.json and .continue-here.md on 2026-05-26_

## Project Reference

**What This Is:** Object Tracking + LSTM Trajectory Prediction — 교통공학·도시개발·통계 연구용 분석 툴

**Core Value:** YOLOv11 기반 객체 추적 + LSTM 궤적 예측에 연구용 분석 모듈(교통·도시·통계)을 통합한 Gradio UI 앱

## Current Position

- **Phase:** 04 of 04 — research-workflow
- **Plan:** 4 of 4 — Complete
- **Status:** All phases done. Awaiting next feature/bug request.

## Progress

`[██████████] 100%` — 4/4 phases complete

## Decisions Made

- **Analyzer 패턴**: 모든 분석 모듈은 `update(tracks, frame)` + `finalize()` 인터페이스 구현 → `pipeline.py`의 `_build_analyzers()`에 등록
- **Gradio UI**: 11개 출력, 아코디언 패널 — 연구자 친화적 구성
- **이중 내보내기**: Excel(openpyxl) + HTML 차트(Plotly)
- **MC-Dropout 불확실성**: CSV에 `pred_std_cx/cy` 컬럼 포함

## Pending Todos

없음.

## Blockers / Concerns

없음.

## Session Continuity

Last session: 2026-05-26T10:38:59.517Z
Stopped at: 4-Phase 확장 전체 완료 후 handoff 저장
Resume file: .planning/phases/04-research-workflow/.continue-here.md
