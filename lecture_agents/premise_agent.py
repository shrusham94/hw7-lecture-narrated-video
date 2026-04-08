from __future__ import annotations

from pathlib import Path
from typing import Any

from google.genai import types

from lecture_agents import json_util
from lecture_agents.llm import generate_json

SYSTEM = """You write a structured lecture premise from slide descriptions only.
Ground claims in the deck. Output JSON with fields you choose but MUST include:
thesis (string), scope (string), learning_objectives (array of strings),
audience (string), key_takeaways (array of strings), prerequisites (array of strings, can be empty),
and session_structure_note (string)."""


def run_premise_agent(slide_description_path: Path, out_json: Path) -> dict[str, Any]:
    raw = slide_description_path.read_text(encoding="utf-8")
    data = generate_json(
        system_instruction=SYSTEM,
        user_parts=[
            types.Part.from_text(
                text="slide_description.json follows. Infer the lecture premise from the whole deck.\n\n"
                + raw
            )
        ],
    )
    out_json.write_text(json_util.dumps_pretty(data), encoding="utf-8")
    return data
