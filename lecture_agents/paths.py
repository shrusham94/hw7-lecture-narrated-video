from __future__ import annotations

from pathlib import Path

# Repo root: parent of lecture_agents/
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_PDF = REPO_ROOT / "Lecture_17_AI_screenplays.pdf"
DEFAULT_TRANSCRIPT = REPO_ROOT / "lecture_transcript.txt"
STYLE_JSON = REPO_ROOT / "style.json"
PROJECTS_DIR = REPO_ROOT / "projects"


def ensure_projects_dir() -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR
