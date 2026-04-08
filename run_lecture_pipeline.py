#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lecture_agents.arc_agent import run_arc_agent
from lecture_agents.narration_agent import run_narration_agent
from lecture_agents.paths import (
    DEFAULT_PDF,
    DEFAULT_TRANSCRIPT,
    PROJECTS_DIR,
    REPO_ROOT,
    STYLE_JSON,
    ensure_projects_dir,
)
from lecture_agents.premise_agent import run_premise_agent
from lecture_agents.rasterize import rasterize_pdf_to_pngs
from lecture_agents.slide_description_agent import run_slide_description_agent
from lecture_agents.style_agent import run_style_agent
from lecture_agents.tts import synthesize_slide_mp3s
from lecture_agents.video import assemble_lecture_mp4

# Load .env from repo root even if the shell cwd is elsewhere
load_dotenv(REPO_ROOT / ".env")
load_dotenv()


def _new_project_dir() -> Path:
    ensure_projects_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = PROJECTS_DIR / f"project_{stamp}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "slide_images").mkdir(exist_ok=True)
    (p / "audio").mkdir(exist_ok=True)
    return p


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Lecture PDF → narrated video pipeline (agents + TTS + ffmpeg).")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Slide deck PDF (default: repo root Lecture_17…)")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT, help="Caption/transcript .txt for style")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Existing projects/project_… folder to resume (skips raster/slide JSON if present unless forced).",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma list: style,raster,slides,premise,arc,narration,tts,video,all",
    )
    parser.add_argument("--force-slides", action="store_true", help="Regenerate slide descriptions even if JSON exists.")
    args = parser.parse_args()

    pdf: Path = args.pdf.resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    steps = {s.strip().lower() for s in args.steps.split(",")}
    if "all" in steps:
        steps = {"style", "raster", "slides", "premise", "arc", "narration", "tts", "video"}

    project_dir = args.project.resolve() if args.project else _new_project_dir()
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "slide_images").mkdir(exist_ok=True)
    (project_dir / "audio").mkdir(exist_ok=True)

    slide_images_dir = project_dir / "slide_images"
    audio_dir = project_dir / "audio"
    slide_desc_path = project_dir / "slide_description.json"
    premise_path = project_dir / "premise.json"
    arc_path = project_dir / "arc.json"
    narr_path = project_dir / "slide_description_narration.json"

    if "style" in steps:
        tp = args.transcript.resolve()
        if not tp.is_file() and not list(tp.parent.glob("lecture_transcript_part*.txt")):
            raise FileNotFoundError(
                f"Transcript missing: add {tp.name} or lecture_transcript_part*.txt (course captions)."
            )
        run_style_agent(tp, STYLE_JSON)

    slide_paths: list[Path] = []
    if "raster" in steps:
        slide_paths = rasterize_pdf_to_pngs(pdf, slide_images_dir)
        print(f"Rasterized {len(slide_paths)} slides → {slide_images_dir}")
    else:
        slide_paths = sorted(slide_images_dir.glob("slide_*.png"))

    if "slides" in steps:
        if not slide_paths:
            slide_paths = rasterize_pdf_to_pngs(pdf, slide_images_dir)
        if args.force_slides or not slide_desc_path.is_file():
            run_slide_description_agent(slide_paths, slide_desc_path)
        else:
            print(f"Keeping existing {slide_desc_path.name} (no --force-slides).")

    if "premise" in steps:
        if not slide_desc_path.is_file():
            raise FileNotFoundError(f"Missing {slide_desc_path}; run slides step first.")
        run_premise_agent(slide_desc_path, premise_path)

    if "arc" in steps:
        if not slide_desc_path.is_file() or not premise_path.is_file():
            raise FileNotFoundError("Need slide_description.json and premise.json before arc.")
        run_arc_agent(premise_path, slide_desc_path, arc_path)

    if "narration" in steps:
        if not STYLE_JSON.is_file():
            raise FileNotFoundError(f"Missing {STYLE_JSON}; run style step first.")
        for req in (premise_path, arc_path, slide_desc_path):
            if not req.is_file():
                raise FileNotFoundError(f"Missing {req}")
        if not slide_paths:
            slide_paths = sorted(slide_images_dir.glob("slide_*.png"))
        run_narration_agent(
            slide_images=slide_paths,
            style_path=STYLE_JSON,
            premise_path=premise_path,
            arc_path=arc_path,
            slide_description_path=slide_desc_path,
            out_json=narr_path,
        )

    if "tts" in steps:
        if not narr_path.is_file():
            raise FileNotFoundError(f"Missing {narr_path}; run narration step first.")
        doc = json.loads(narr_path.read_text(encoding="utf-8"))
        slides = doc.get("slides") or []
        synthesize_slide_mp3s(slides, audio_dir)

    if "video" in steps:
        if not slide_paths:
            slide_paths = sorted(slide_images_dir.glob("slide_*.png"))
        out_name = pdf.name.rsplit(".", 1)[0] + ".mp4"
        out_mp4 = project_dir / out_name
        assemble_lecture_mp4(slide_images_dir=slide_images_dir, audio_dir=audio_dir, out_mp4=out_mp4)
        print(f"Wrote {out_mp4}")

    print(f"Project folder: {project_dir}")


if __name__ == "__main__":
    main()
