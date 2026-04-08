from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.genai import types
from tqdm import tqdm

from lecture_agents import json_util
from lecture_agents.llm import generate_json, part_from_image_path

BASE_SYSTEM = """You write spoken lecture narration for a slide deck video.
Match the instructor's speaking style using style.json (tone, fillers guidance, pacing).
Must stay aligned with premise.json + arc.json + this slide's description.
Do not claim to see live chat. Keep each narration tight for TTS (no stage directions).

Output JSON: {"slide_index": <int>, "narration": <string>}"""


def run_narration_agent(
    *,
    slide_images: list[Path],
    style_path: Path,
    premise_path: Path,
    arc_path: Path,
    slide_description_path: Path,
    out_json: Path,
) -> dict[str, Any]:
    style_txt = style_path.read_text(encoding="utf-8")
    premise_txt = premise_path.read_text(encoding="utf-8")
    arc_txt = arc_path.read_text(encoding="utf-8")
    slide_doc = json.loads(slide_description_path.read_text(encoding="utf-8"))

    slides_in = slide_doc.get("slides") or []
    by_idx = {int(s["slide_index"]): s for s in slides_in if "slide_index" in s}

    prior_narrations: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for idx, img in enumerate(tqdm(slide_images, desc="Narrations"), start=1):
        meta = by_idx.get(idx, {})
        desc_text = meta.get("description", "")
        key_points = meta.get("key_points") or []

        prior_blob = json_util.dumps_pretty({"prior_slide_narrations": prior_narrations})
        context_pack = json_util.dumps_pretty(
            {
                "current_slide_index": idx,
                "current_slide_description": desc_text,
                "current_key_points": key_points,
            }
        )

        title_rules = ""
        if idx == 1:
            title_rules = (
                "\nSPECIAL TITLE SLIDE RULE: The speaker MUST introduce themselves plausibly "
                "as the course instructor (first-person). Give a short overview of the lecture topic "
                "and why it matters. Keep it natural and aligned with style.json.\n"
            )

        prompt = (
            f"{title_rules}\n"
            "Use these full-context documents:\n"
            "style.json:\n"
            f"{style_txt}\n\n"
            "premise.json:\n"
            f"{premise_txt}\n\n"
            "arc.json:\n"
            f"{arc_txt}\n\n"
            "slide_description.json excerpt for this slide only is in context_pack below.\n"
            f"{context_pack}\n\n"
            "All prior slide narrations (to avoid repetition):\n"
            f"{prior_blob}\n\n"
            f"Now write narration for slide_index={idx} only."
        )

        data = generate_json(
            system_instruction=BASE_SYSTEM,
            user_parts=[part_from_image_path(img), types.Part.from_text(text=prompt)],
            temperature=0.55,
        )
        if int(data.get("slide_index", -1)) != idx:
            data["slide_index"] = idx
        narration = (data.get("narration") or "").strip()
        entry = {
            "slide_index": idx,
            "image": img.name,
            "title_guess": meta.get("title_guess"),
            "description": desc_text,
            "key_points": key_points,
            "narration": narration,
        }
        combined.append(entry)
        prior_narrations.append({"slide_index": idx, "narration": narration})

    doc: dict[str, Any] = {
        "schema_version": 1,
        "slide_count": len(combined),
        "slides": combined,
    }
    out_json.write_text(json_util.dumps_pretty(doc), encoding="utf-8")
    return doc
