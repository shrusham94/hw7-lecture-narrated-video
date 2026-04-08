from __future__ import annotations

from pathlib import Path
from typing import Any

from google.genai import types

from lecture_agents import json_util
from lecture_agents.llm import generate_json
from lecture_agents.paths import STYLE_JSON

PART_GLOB = "lecture_transcript_part*.txt"

SYSTEM = """You are an analyst of spoken lecture style.
From the raw instructor transcript, infer how this person actually talks when teaching.
Output structured JSON for downstream narration generation. Be concrete and quote short examples
where helpful (paraphrase is fine). Stay faithful to the transcript; do not invent unrelated traits."""


def load_transcript_text(transcript_path: Path) -> str:
    if transcript_path.is_file():
        return transcript_path.read_text(encoding="utf-8", errors="replace")
    parts = sorted(transcript_path.parent.glob(PART_GLOB))
    if not parts:
        raise FileNotFoundError(
            f"No transcript at {transcript_path} and no {PART_GLOB} beside it."
        )
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in parts)


def run_style_agent(transcript_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    text = load_transcript_text(transcript_path)
    dest = out_path or STYLE_JSON
    source_label = (
        transcript_path.name
        if transcript_path.is_file()
        else "lecture_transcript_part*.txt (concatenated)"
    )
    payload = {
        "topic_hint": (
            "Use this transcript only to model delivery; slide content may differ "
            "(e.g. another lecture)."
        ),
        "transcript_source": source_label,
        "transcript_excerpt_chars": min(len(text), 120_000),
        "transcript": text[:120_000],
    }
    user = types.Part.from_text(
        text=(
            "Analyze the instructor's speaking style from this transcript and fill:\n"
            "- tone: overall tone\n"
            "- register: formality (e.g. casual MBA classroom)\n"
            "- pacing_notes: fast/slow, digressions, callbacks\n"
            "- fillers_and_discourse_markers: list of recurring fillers ('um', 'like', 'right?')\n"
            "- rhetorical_moves: how they frame ideas (problem/solution, analogy, challenge)\n"
            "- humor_and_asides: how jokes or tangents show up\n"
            "- audience_interaction: chat, questions, reassurance patterns\n"
            "- emphasis_patterns: repetition, contrast, spoilers/warnings\n"
            "- vocabulary_notes: jargon level, catchphrases\n"
            "- narration_guidance: bullet list telling a TTS scriptwriter how to sound like this instructor\n"
            "- avoid: things the narration must NOT do (e.g. swear if instructor avoids it)\n"
        )
    )
    data = generate_json(
        system_instruction=SYSTEM,
        user_parts=[
            user,
            types.Part.from_text(text=json_util.dumps_pretty(payload)),
        ],
    )
    dest.write_text(json_util.dumps_pretty(data), encoding="utf-8")
    return data
