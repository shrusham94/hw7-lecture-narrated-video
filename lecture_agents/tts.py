from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from lecture_agents.paths import REPO_ROOT


async def _edge_tts_save(text: str, out_mp3: Path, voice: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_mp3.as_posix())


def synthesize_slide_mp3s(
    narrations: list[dict[str, Any]],
    audio_dir: Path,
    *,
    edge_voice: str | None = None,
) -> None:
    """
    Write audio/slide_NNN.mp3 for each slide using ElevenLabs if ELEVENLABS_API_KEY is set,
    otherwise Microsoft Edge TTS (no API key).
    """
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
    audio_dir.mkdir(parents=True, exist_ok=True)
    key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    voice_id = (os.environ.get("ELEVENLABS_VOICE_ID") or "").strip()

    if key and voice_id:
        _elevenlabs_batch(narrations, audio_dir, api_key=key, voice_id=voice_id)
        return

    voice = edge_voice or (os.environ.get("EDGE_TTS_VOICE") or "en-US-GuyNeural").strip()

    async def _run_all() -> None:
        sem = asyncio.Semaphore(4)

        async def one(item: dict[str, Any]) -> None:
            idx = int(item["slide_index"])
            text = (item.get("narration") or "").strip()
            if not text:
                text = "This slide has no narration text."
            out = audio_dir / f"slide_{idx:03d}.mp3"
            async with sem:
                await _edge_tts_save(text, out, voice)

        await asyncio.gather(*(one(s) for s in narrations))

    asyncio.run(_run_all())


def _elevenlabs_batch(
    narrations: list[dict[str, Any]],
    audio_dir: Path,
    *,
    api_key: str,
    voice_id: str,
) -> None:
    url_base = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Accept": "audio/mpeg"}
    settings = {
        "model_id": (os.environ.get("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2").strip(),
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.75},
    }

    for item in tqdm(narrations, desc="TTS (ElevenLabs)"):
        idx = int(item["slide_index"])
        text = (item.get("narration") or "").strip()
        if not text:
            text = "This slide has no narration text."
        out = audio_dir / f"slide_{idx:03d}.mp3"
        chunks = _chunk_text(text, max_chars=8000)
        if len(chunks) == 1:
            _write_elevenlabs_chunk(url_base, headers, settings, chunks[0], out)
            continue
        parts: list[Path] = []
        for c_i, chunk in enumerate(chunks):
            part = audio_dir / f"slide_{idx:03d}_part_{c_i:03d}.mp3"
            _write_elevenlabs_chunk(url_base, headers, settings, chunk, part)
            parts.append(part)
        _ffmpeg_concat_audio(parts, out)
        for p in parts:
            p.unlink(missing_ok=True)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            continue
        add = len(p) + 1
        if size + add > max_chars and buf:
            out.append("\n".join(buf))
            buf = [p]
            size = len(p)
        else:
            buf.append(p)
            size += add
    if buf:
        out.append("\n".join(buf))
    return out


def _write_elevenlabs_chunk(
    url: str,
    headers: dict[str, str],
    payload_template: dict[str, Any],
    text: str,
    out: Path,
) -> None:
    body = dict(payload_template)
    body["text"] = text
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        out.write_bytes(r.content)


def _ffmpeg_concat_audio(parts: list[Path], out: Path) -> None:
    import subprocess
    import tempfile

    lines = "\n".join(f"file '{p.as_posix()}'" for p in parts)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(lines)
        lst = f.name
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            lst,
            "-c",
            "copy",
            out.as_posix(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        try:
            os.unlink(lst)
        except OSError:
            pass
