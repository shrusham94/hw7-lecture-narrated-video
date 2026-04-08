from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from lecture_agents.json_util import extract_json_object
from lecture_agents.paths import REPO_ROOT

DEFAULT_MODEL = "gemini-2.5-flash"


def _api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env for lecture agents.")
    return key


def model_name() -> str:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
    return (os.environ.get("LECTURE_GEMINI_MODEL") or DEFAULT_MODEL).strip()


def client() -> genai.Client:
    return genai.Client(api_key=_api_key())


def generate_text(
    *,
    system_instruction: str,
    user_parts: list[types.Part],
    temperature: float = 0.4,
    response_mime_type: str | None = None,
    max_output_tokens: int | None = None,
) -> str:
    c = client()
    cfg_kw: dict[str, Any] = {
        "systemInstruction": system_instruction,
        "temperature": temperature,
    }
    if response_mime_type:
        cfg_kw["response_mime_type"] = response_mime_type
    if max_output_tokens is not None:
        cfg_kw["max_output_tokens"] = max_output_tokens
    cfg = types.GenerateContentConfig(**cfg_kw)
    resp = c.models.generate_content(
        model=model_name(),
        contents=[types.Content(role="user", parts=user_parts)],
        config=cfg,
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Empty model response (lecture agents).")
    return text


def generate_json(
    *,
    system_instruction: str,
    user_parts: list[types.Part],
    temperature: float = 0.3,
) -> dict[str, Any]:
    sys = system_instruction + "\n\nRespond with a single JSON object only."
    last_err: BaseException | None = None
    for attempt in range(4):
        try:
            raw = generate_text(
                system_instruction=sys,
                user_parts=user_parts,
                temperature=temperature,
                response_mime_type="application/json",
                max_output_tokens=8192,
            )
            return extract_json_object(raw)
        except (RuntimeError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            if attempt < 3:
                time.sleep(1.5 * (2**attempt))
                continue
            raise RuntimeError(f"Lecture agent JSON failed after retries: {e}") from e


def part_from_image_path(path: Path) -> types.Part:
    data = path.read_bytes()
    return types.Part.from_bytes(data=data, mime_type="image/png")
