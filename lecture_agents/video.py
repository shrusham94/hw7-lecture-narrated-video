from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


def assemble_lecture_mp4(
    *,
    slide_images_dir: Path,
    audio_dir: Path,
    out_mp4: Path,
    work_dir: Path | None = None,
) -> Path:
    """
    For each slide N, mux slide_NNN.png with slide_NNN.mp3 using -shortest (duration follows audio).
    Concatenate segments into one MP4 via ffmpeg concat demuxer.
    """
    images = sorted(slide_images_dir.glob("slide_*.png"))
    if not images:
        raise FileNotFoundError(f"No PNGs in {slide_images_dir}")

    tmp_root = work_dir or Path(tempfile.mkdtemp(prefix="lecture_vid_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []

    for img in tqdm(images, desc="Mux segments"):
        name = img.stem  # slide_001
        idx = name.split("_")[-1]
        mp3 = audio_dir / f"slide_{idx}.mp3"
        if not mp3.is_file():
            raise FileNotFoundError(f"Missing audio for {name}: {mp3}")
        seg = tmp_root / f"segment_{idx}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            img.as_posix(),
            "-i",
            mp3.as_posix(),
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            seg.as_posix(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        segments.append(seg)

    list_path = tmp_root / "concat.txt"
    list_path.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments) + "\n", encoding="utf-8")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path.as_posix(),
        "-c",
        "copy",
        out_mp4.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    if work_dir is None:
        for p in segments:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            list_path.unlink()
        except OSError:
            pass
        try:
            os.rmdir(tmp_root)
        except OSError:
            pass

    return out_mp4
