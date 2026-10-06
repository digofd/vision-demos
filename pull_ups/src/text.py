"""TrueType text rendering for video overlays via Pillow, since OpenCV only has Hershey fonts.

Callers fall back to Hershey if Pillow or a usable face is missing.
"""

from __future__ import annotations

from pathlib import Path

# Ordered preference of (path, face index within a .ttc, name).
# Indices are verified: e.g. Avenir Next face 1 is Bold Italic, not Bold.
FONT_CANDIDATES: tuple[tuple[str, int, str], ...] = (
    ("/System/Library/Fonts/Avenir Next.ttc", 0, "Avenir Next Bold"),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1, "Helvetica Neue Bold"),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0, "Arial Bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, "DejaVu Sans Bold"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0, "Liberation Sans Bold"),
)


def _matplotlib_dejavu() -> tuple[str, int, str] | None:
    """DejaVu Sans Bold ships inside matplotlib, the portable last resort."""
    try:
        import matplotlib
    except ImportError:
        return None
    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return (str(path), 0, "DejaVu Sans Bold (bundled)") if path.is_file() else None


def resolve_font(spec: str = "auto", index: int | None = None) -> tuple[str, int, str] | None:
    """Pick a font face, or None for the OpenCV fallback.

    ``spec`` is "auto", "opencv" (force Hershey), or a path to a .ttf/.otf/.ttc.
    """
    try:
        from PIL import ImageFont  # noqa: F401
    except ImportError:
        return None

    if spec == "opencv":
        return None

    if spec != "auto":
        path = Path(spec).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"PANEL_FONT points at a missing file: {path}")
        return (str(path), index or 0, path.stem)

    for path, idx, name in FONT_CANDIDATES:
        if Path(path).is_file():
            return (path, index if index is not None else idx, name)
    return _matplotlib_dejavu()
