from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def rasterize_pdf_to_pngs(pdf_path: Path, out_dir: Path, zoom: float = 2.0) -> list[Path]:
    """
    Render each PDF page to slide_XXX.png under out_dir.
    Returns paths sorted by slide index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(zoom, zoom)
    paths: list[Path] = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            name = f"slide_{i + 1:03d}.png"
            fp = out_dir / name
            pix.save(fp.as_posix())
            paths.append(fp)
    finally:
        doc.close()
    return paths
