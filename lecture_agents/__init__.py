"""Lecture deck -> narrated video pipeline (HW agents)."""

from dotenv import load_dotenv

from lecture_agents.paths import REPO_ROOT

# Repo-root .env (GEMINI_API_KEY / GOOGLE_API_KEY) before any agent runs, any cwd
load_dotenv(REPO_ROOT / ".env")
load_dotenv()
