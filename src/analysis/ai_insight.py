"""Claude API를 이용해 분석 결과(last_analysis/last_result)를 한국어 해설로 요약한다.

이 모듈은 Gradio·네트워크 의존성이 없는 build_analysis_summary()와, anthropic SDK를
호출하는 generate_insight()로 나뉜다. 전자는 process_videos() 내부에서 즉시 호출해도
지연이 없고, 후자는 "AI 인사이트 생성" 버튼 클릭 시에만 호출된다.
"""
import json

from .session import _NumpyEncoder

DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_CHOICES = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
MAX_TOKENS = 2048


class AIInsightError(Exception):
    """사용자에게 그대로 보여줄 수 있는 실패(키 누락/인증/한도/네트워크/거부).

    app.py는 이 예외만 gr.Warning으로 변환하고, 그 외 예외는 버그로 취급해 그대로
    전파한다.
    """


# ── 데이터 축소 (토큰 절약) ──────────────────────────────────────────────────

def _summarize_counting_lines(counting_lines: list) -> list:
    out = []
    for line in counting_lines or []:
        out.append({
            "label": line.get("label"),
            "counts": line.get("counts", {}),
            "flow_rates_veh_hr": line.get("flow_rates_veh_hr", {}),
            "headway": line.get("headway", {}),
            "duration_s": line.get("duration_s"),
            "total_crossings": sum((line.get("counts") or {}).values()),
        })
    return out


def _summarize_speed(speed_data: dict) -> dict:
    per_class = (speed_data or {}).get("per_class", {}) or {}
    return {"per_class": per_class}


def _summarize_od_matrix(od_data: dict) -> dict:
    if not od_data or "matrix_df" not in od_data:
        return {}
    df = od_data["matrix_df"]
    zone_names = od_data.get("zone_names", list(df.index))
    matrix = {}
    for origin in df.index:
        row = {
            dest: int(df.loc[origin, dest])
            for dest in df.columns
            if df.loc[origin, dest] != 0
        }
        if row:
            matrix[origin] = row
    return {"zone_names": list(zone_names), "od_counts": matrix}


def _summarize_zone_analysis(zone_data: dict) -> dict:
    summaries = (zone_data or {}).get("zone_summaries") or []
    return {"zone_summaries": summaries}


def _summarize_track_summary(track_summary: list) -> dict:
    if not track_summary:
        return {"total_tracks": 0, "by_class": {}}

    by_class: dict = {}
    for row in track_summary:
        by_class.setdefault(row.get("class", "unknown"), []).append(row)

    dist_key = "distance_m" if any("distance_m" in r for r in track_summary) else "distance_px"

    out = {}
    for cls, rows in by_class.items():
        durations = [r["duration_frames"] for r in rows if r.get("duration_frames") is not None]
        distances = [r[dist_key] for r in rows if r.get(dist_key) is not None]
        speeds = [r["mean_speed_kmh"] for r in rows if r.get("mean_speed_kmh") is not None]
        out[cls] = {
            "count": len(rows),
            "mean_duration_frames": round(sum(durations) / len(durations), 1) if durations else None,
            "mean_distance": round(sum(distances) / len(distances), 2) if distances else None,
            "distance_unit": "m" if dist_key == "distance_m" else "px",
            "mean_speed_kmh": round(sum(speeds) / len(speeds), 1) if speeds else None,
        }
    return {"total_tracks": len(track_summary), "by_class": out}


def _summarize_congestion(congestion_data: dict) -> dict:
    events = (congestion_data or {}).get("events") or []
    total = len(events)
    if total > 10:
        events = sorted(events, key=lambda e: e.get("duration_s", 0), reverse=True)[:10]
    return {"events": events, "total_events": total}


def build_analysis_summary(last_analysis: dict, last_result: dict, enabled_flags: dict) -> dict:
    """last_analysis/last_result로부터 LLM에 보낼 작은 JSON-safe 요약 dict를 만든다.

    enabled_flags는 process_videos()의 로컬 변수(enable_traffic 등)를 그대로 받는다 —
    토글이 꺼진 것과 켜졌지만 결과가 0인 것은 last_analysis 모양만으로 구분할 수
    없기 때문이다.
    """
    last_analysis = last_analysis or {}
    last_result = last_result or {}
    stats = last_result.get("stats", {}) or {}

    summary = {
        "meta": {
            "total_frames": stats.get("total_frames"),
            "total_tracks": stats.get("total_tracks"),
            "duration_s": round(last_result.get("duration_s", 0) or 0, 1),
            "enabled_analyses": {k: bool(v) for k, v in (enabled_flags or {}).items()},
        },
    }

    if enabled_flags.get("enable_traffic"):
        summary["counting_lines"] = _summarize_counting_lines(last_analysis.get("counting_lines", []))
        summary["congestion"] = _summarize_congestion(last_analysis.get("congestion", {}))

    if enabled_flags.get("enable_speed"):
        summary["speed"] = _summarize_speed(last_analysis.get("speed", {}))

    if enabled_flags.get("enable_od"):
        summary["od_matrix"] = _summarize_od_matrix(last_analysis.get("od_matrix", {}))

    if enabled_flags.get("enable_urban"):
        summary["zone_analysis"] = _summarize_zone_analysis(last_analysis.get("zone_analysis", {}))

    summary["track_summary"] = _summarize_track_summary(last_result.get("track_summary", []))

    return summary


# ── 프롬프트 구성 ────────────────────────────────────────────────────────────

def _build_system_prompt(summary: dict) -> str:
    sections = ["핵심 요약 — 전체 분석 결과를 2~3문장으로 요약"]

    if "counting_lines" in summary:
        sections.append("교통 흐름 — 감지선별 통과량, 유량(대/시), 헤드웨이 분석")
    if "congestion" in summary:
        sections.append("혼잡 구간 — 혼잡 이벤트의 발생 시점, 지속시간, 원인 추정")
    if "speed" in summary:
        sections.append("속도 분포 — 클래스별 속도 통계에서 주목할 패턴이나 이상치")
    if "od_matrix" in summary:
        sections.append("OD 흐름 — 존 간 이동 패턴 중 특이한 흐름")
    if "zone_analysis" in summary:
        sections.append("존별 특이사항 — 체류시간 및 밀도가 높거나 낮은 존, 피크 시간대")

    sections.append("트랙 요약 해석 — 클래스별 트랙 수, 이동거리, 속도에서 보이는 경향")
    sections.append("권장사항 — 데이터에 기반한 실행 가능한 제안 2~4개")

    section_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sections))

    role = "당신은 차량 및 보행자 영상 추적 데이터를 해석하는 교통 및 도시공간 분석 전문가입니다."
    context = (
        "아래 사용자 메시지에는 YOLO 객체 탐지 + ByteTrack 추적 + LSTM 경로 예측 파이프라인이 "
        "하나의 영상을 처리한 결과를 요약한 JSON 데이터가 주어집니다. 이 데이터를 바탕으로 "
        "한국어로 구조화된 분석 리포트를 작성하세요."
    )
    format_instruction = "리포트는 다음 섹션을 마크다운 헤더(##)로 구성하세요:"
    notes = (
        "작성 시 주의사항:\n"
        "- 반드시 주어진 JSON 데이터에 근거해서만 서술하세요. 데이터에 없는 수치를 추측하거나 지어내지 마세요.\n"
        "- 데이터가 비어 있거나 패턴이 없는 섹션은 억지로 분석하지 말고 특이사항 없음을 짧게 언급하세요.\n"
        "- 숫자는 JSON 값을 그대로 인용하고 단위(km/h, 대/시, m, s 등)를 함께 표기하세요.\n"
        "- 각 섹션을 2~5문장 또는 3~5개 불릿으로 간결하게 작성하세요.\n"
        "- track_summary는 클래스별 집계 통계이며 개별 트랙 데이터가 아님을 인지하고 해석하세요.\n"
    )

    return f"{role}\n\n{context}\n\n{format_instruction}\n\n{section_list}\n\n{notes}"


def _build_user_message(summary: dict) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


# ── Claude API 호출 ─────────────────────────────────────────────────────────

def generate_insight(summary: dict, model: str = DEFAULT_MODEL, api_key: str | None = None) -> str:
    """summary를 Claude API에 보내 한국어 해설 텍스트를 받아온다.

    실패 시 사용자에게 보여줄 수 있는 한국어 메시지를 담은 AIInsightError를 발생시킨다.
    예상치 못한 예외(진짜 버그)는 그대로 전파된다.
    """
    import os

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIInsightError(
            "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 ANTHROPIC_API_KEY=sk-ant-... 를 추가하거나 "
            "환경 변수로 설정한 뒤 다시 시도해주세요."
        )

    import anthropic  # 지연 임포트: anthropic 패키지를 선택적 의존성으로 취급

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_system_prompt(summary)
    user_message = _build_user_message(summary)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        raise AIInsightError("Claude API 인증에 실패했습니다. ANTHROPIC_API_KEY가 올바른지 확인해주세요.")
    except anthropic.RateLimitError:
        raise AIInsightError("Claude API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except anthropic.APIStatusError as e:
        raise AIInsightError(f"Claude API 오류가 발생했습니다 (status={e.status_code}). 잠시 후 다시 시도해주세요.")
    except anthropic.APIConnectionError:
        raise AIInsightError("Claude API 서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.")

    if response.stop_reason == "refusal":
        raise AIInsightError("Claude가 이 요청에 대한 응답을 거부했습니다. 데이터를 확인 후 다시 시도해주세요.")

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise AIInsightError("Claude로부터 빈 응답을 받았습니다. 다시 시도해주세요.")
    return text
