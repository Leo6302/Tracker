"""분석 결과(last_analysis/last_result)를 LLM(Claude API 또는 로컬 Ollama)으로 한국어
해설로 요약한다.

이 모듈은 Gradio·네트워크 의존성이 없는 build_analysis_summary()와, 실제 LLM을 호출하는
generate_insight()(provider에 따라 _generate_with_claude/_generate_with_ollama로 분기)로
나뉜다. 전자는 process_videos() 내부에서 즉시 호출해도 지연이 없고, 후자는 "AI 인사이트
생성" 버튼 클릭 시에만 호출된다.

Claude는 구조화 출력(JSON 스키마)으로 report_markdown + highlights(차트/표에 표시할 메모)를
함께 받는다. 로컬 모델은 JSON 형식 준수가 불안정할 수 있어 텍스트 리포트만 받고
highlights는 항상 빈 리스트다 — generate_insight()는 두 경로 모두
{"report_markdown": str, "highlights": list} 형태로 반환을 통일한다.
"""
import json

from .session import _NumpyEncoder

DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_CHOICES = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
MAX_TOKENS = 3072  # report_markdown + highlights JSON을 함께 받으므로 기존 2048보다 여유를 둠

DEFAULT_OLLAMA_MODEL = "exaone3.5:7.8b"
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
                    "target": {"type": "string"},
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

def _report_sections(summary: dict) -> list:
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
    return sections


def _prompt_preamble(summary: dict) -> tuple:
    sections = _report_sections(summary)
    section_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sections))

    role = "당신은 차량 및 보행자 영상 추적 데이터를 해석하는 교통 및 도시공간 분석 전문가입니다."
    context = (
        "아래 사용자 메시지에는 YOLO 객체 탐지 + ByteTrack 추적 + LSTM 경로 예측 파이프라인이 "
        "하나의 영상을 처리한 결과를 요약한 JSON 데이터가 주어집니다. 이 데이터를 바탕으로 "
        "한국어로 구조화된 분석 리포트를 작성하세요."
    )
    notes = (
        "작성 시 주의사항:\n"
        "- 반드시 주어진 JSON 데이터에 근거해서만 서술하세요. 데이터에 없는 수치를 추측하거나 지어내지 마세요.\n"
        "- 데이터가 비어 있거나 패턴이 없는 섹션은 억지로 분석하지 말고 특이사항 없음을 짧게 언급하세요.\n"
        "- 숫자는 JSON 값을 그대로 인용하고 단위(km/h, 대/시, m, s 등)를 함께 표기하세요.\n"
        "- 각 섹션을 2~5문장 또는 3~5개 불릿으로 간결하게 작성하세요.\n"
        "- track_summary는 클래스별 집계 통계이며 개별 트랙 데이터가 아님을 인지하고 해석하세요.\n"
    )
    return role, context, section_list, notes


def _build_text_system_prompt(summary: dict) -> str:
    """로컬 Ollama용 — 마크다운 리포트 텍스트만 요청(JSON 래핑 없음)."""
    role, context, section_list, notes = _prompt_preamble(summary)
    format_instruction = "리포트는 다음 섹션을 마크다운 헤더(##)로 구성하세요:"
    return f"{role}\n\n{context}\n\n{format_instruction}\n\n{section_list}\n\n{notes}"


def _build_structured_system_prompt(summary: dict) -> str:
    """Claude용 — report_markdown(위와 동일한 리포트) + highlights 배열을 함께 요청."""
    role, context, section_list, notes = _prompt_preamble(summary)
    format_instruction = (
        "응답은 report_markdown과 highlights 두 필드를 가진 JSON으로 작성하세요. "
        "report_markdown에는 다음 섹션을 마크다운 헤더(##)로 구성한 리포트를 담으세요:"
    )
    highlights_instruction = (
        "추가로 highlights 배열에는, 데이터에 실제로 존재하는 라벨(클래스명/존 이름/감지선 이름)을 "
        "정확히 그대로 사용해 가장 주목할 만한 데이터 포인트 3~6개를 뽑아 담으세요. "
        "각 항목은 연구 방향을 제시할 수 있는 이상치·특이 패턴·실행 가능한 시사점이어야 하며, "
        "category는 해당 데이터가 속한 항목"
        "(counting_lines/speed/zone_analysis/od_matrix/congestion/track_summary) 중 "
        "이 요약에 실제로 존재하는 것만 사용하세요. target에 존재하지 않는 라벨을 지어내지 마세요."
    )
    return f"{role}\n\n{context}\n\n{format_instruction}\n\n{section_list}\n\n{notes}\n\n{highlights_instruction}"


def _build_user_message(summary: dict) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


def render_insight_markdown(result: dict) -> str:
    """generate_insight()의 결과를 패널에 표시할 최종 마크다운 문자열로 합친다.

    highlights가 있으면(Claude 경로) 맨 위에 강조점 블록을 추가하고, 없으면(Ollama
    경로, 또는 Claude가 하이라이트를 찾지 못한 경우) report_markdown만 그대로 반환한다.
    """
    highlights = result.get("highlights") or []
    report = result.get("report_markdown", "")
    if not highlights:
        return report

    lines = ["## 🔍 주목할 포인트 (연구 방향성 힌트)", ""]
    ordered = sorted(highlights, key=lambda h: 0 if h.get("importance") == "high" else 1)
    for h in ordered:
        mark = "🔴" if h.get("importance") == "high" else "🔵"
        label = _CATEGORY_LABELS.get(h.get("category"), h.get("category", ""))
        lines.append(f"- {mark} **[{label}] {h.get('target', '')}** — {h.get('note', '')}")
    lines.append("\n---\n")
    return "\n".join(lines) + report


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

def generate_insight(
    summary: dict,
    provider: str = "claude",
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> dict:
    """summary를 LLM에 보내 해설을 받아온다.

    provider="claude"면 Anthropic API를 구조화 출력으로 호출해 {"report_markdown",
    "highlights"}를 받고, provider="ollama"면 로컬 Ollama 서버를 호출해 텍스트
    리포트만 받는다(highlights는 항상 []). 두 경로 모두 같은 모양의 dict를 반환한다.

    실패 시 사용자에게 보여줄 수 있는 한국어 메시지를 담은 AIInsightError를 발생시킨다.
    예상치 못한 예외(진짜 버그)는 그대로 전파된다.
    """
    if provider == "ollama":
        return _generate_with_ollama(summary, ollama_model, ollama_host)
    return _generate_with_claude(summary, model, api_key)


def _generate_with_claude(summary: dict, model: str, api_key: str | None) -> dict:
    import os

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIInsightError(
            "Anthropic API 키가 없습니다. 위 'Anthropic API 키' 입력란에 붙여넣거나 "
            ".env 파일에 ANTHROPIC_API_KEY=sk-ant-... 를 추가한 뒤 다시 시도해주세요."
        )

    import anthropic  # 지연 임포트: anthropic 패키지를 선택적 의존성으로 취급

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=_build_structured_system_prompt(summary),
            messages=[{"role": "user", "content": _build_user_message(summary)}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
    except anthropic.AuthenticationError:
        raise AIInsightError("Claude API 인증에 실패했습니다. API 키가 올바른지 확인해주세요.")
    except anthropic.RateLimitError:
        raise AIInsightError("Claude API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except anthropic.APIStatusError as e:
        raise AIInsightError(f"Claude API 오류가 발생했습니다 (status={e.status_code}). 잠시 후 다시 시도해주세요.")
    except anthropic.APIConnectionError:
        raise AIInsightError("Claude API 서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.")

    if response.stop_reason == "refusal":
        raise AIInsightError("Claude가 이 요청에 대한 응답을 거부했습니다. 데이터를 확인 후 다시 시도해주세요.")

    text = next((block.text for block in response.content if block.type == "text"), "")
    if not text:
        raise AIInsightError("Claude로부터 빈 응답을 받았습니다. 다시 시도해주세요.")

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise AIInsightError("Claude 응답을 해석할 수 없습니다. 다시 시도해주세요.")
    result.setdefault("highlights", [])
    return result


def _generate_with_ollama(summary: dict, model: str, host: str) -> dict:
    import httpx  # 지연 임포트: anthropic의 전이 의존성이라 항상 설치돼 있지만 일관성 유지

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_text_system_prompt(summary)},
            {"role": "user", "content": _build_user_message(summary)},
        ],
        "stream": False,
    }

    try:
        resp = httpx.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=120.0)
    except httpx.ConnectError:
        raise AIInsightError(
            f"로컬 Ollama 서버({host})에 연결할 수 없습니다. Ollama를 설치·실행한 뒤 다시 시도해주세요."
        )
    except httpx.TimeoutException:
        raise AIInsightError("Ollama 응답 시간이 초과되었습니다. 더 작은 모델을 사용해보세요.")

    if resp.status_code == 404:
        raise AIInsightError(
            f"Ollama 모델 '{model}'을 찾을 수 없습니다. "
            f"터미널에서 `ollama pull {model}` 실행 후 다시 시도해주세요."
        )
    if resp.status_code != 200:
        raise AIInsightError(f"Ollama 오류가 발생했습니다 (status={resp.status_code}).")

    text = (resp.json().get("message") or {}).get("content", "").strip()
    if not text:
        raise AIInsightError("Ollama로부터 빈 응답을 받았습니다. 다시 시도해주세요.")
    return {"report_markdown": text, "highlights": []}
