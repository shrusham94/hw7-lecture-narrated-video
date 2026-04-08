from __future__ import annotations

from pathlib import Path
from typing import Any

from google.genai import types
from tqdm import tqdm

from lecture_agents import json_util
from lecture_agents.llm import generate_json, part_from_image_path

SYSTEM = """You describe presentation slides for a narrated lecture video.
Given the current slide image and accurate prior slide descriptions (in order), write a concise
description of ONLY the current slide: visible title, bullets, diagrams, and how it advances the deck.
Do not repeat prior slides; use prior context only for coherence.
Output JSON: {"slide_index": <int>, "title_guess": <str or null>, "description": <str>, "key_points": [<str>]}
slide_index must match the prompt."""


def run_slide_description_agent(slide_images: list[Path], out_json: Path) -> dict[str, Any]:
    prior: list[dict[str, Any]] = []
    slides_out: list[dict[str, Any]] = []
    for idx, img in enumerate(tqdm(slide_images, desc="Slide descriptions"), start=1):
        prior_blob = json_util.dumps_pretty({"prior_slide_descriptions": prior})
        prompt = (
            f"slide_index={idx}.\n"
            "Prior slides (most recent last):\n"
            f"{prior_blob}\n"
            "Now describe ONLY this slide image."
        )
        data = generate_json(
            system_instruction=SYSTEM,
            user_parts=[part_from_image_path(img), types.Part.from_text(text=prompt)],
        )
        if int(data.get("slide_index", -1)) != idx:
            data["slide_index"] = idx
        slides_out.append(
            {
                "slide_index": idx,
                "image": img.name,
                "title_guess": data.get("title_guess"),
                "description": data.get("description", ""),
                "key_points": data.get("key_points") or [],
            }
        )
        prior.append(
            {
                "slide_index": idx,
                "title_guess": data.get("title_guess"),
                "description": data.get("description", ""),
                "key_points": data.get("key_points") or [],
            }
        )
    doc: dict[str, Any] = {
        "schema_version": 1,
        "slide_count": len(slides_out),
        "slides": slides_out,
    }
    out_json.write_text(json_util.dumps_pretty(doc), encoding="utf-8")
    return doc
