from __future__ import annotations

from pathlib import Path
from typing import Any

from google.genai import types

from lecture_agents import json_util
from lecture_agents.llm import generate_json

SYSTEM = """You design a teaching arc consistent with the premise and slide descriptions.
Output JSON with: flow_summary (string), acts_or_phases (array of objects with label, goal, slide_range_hint),
progression_notes (string), rhetorical_strategy (string), transitions (array of strings).
Slide_range_hint can be like "1-3" or approximate; tie phases to actual slide themes."""


def run_arc_agent(premise_path: Path, slide_description_path: Path, out_json: Path) -> dict[str, Any]:
    premise = premise_path.read_text(encoding="utf-8")
    slides = slide_description_path.read_text(encoding="utf-8")
    data = generate_json(
        system_instruction=SYSTEM,
        user_parts=[
            types.Part.from_text(
                text="premise.json:\n"
                + premise
                + "\n\nslide_description.json:\n"
                + slides
                + "\n\nProduce arc.json."
            )
        ],
    )
    out_json.write_text(json_util.dumps_pretty(data), encoding="utf-8")
    return data
