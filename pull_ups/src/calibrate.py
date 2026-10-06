"""Learning what *this* athlete's full range of motion looks like.

Run `calibrate.py` once on a good set; later runs are graded against its measured
hang angle, top angle and head clearance instead of config.py constants. Hang
minus top is the sweep of a full pull. See how-it-works.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROFILE_VERSION = 1


@dataclass(frozen=True)
class Stat:
    """One measurement across the reference reps."""

    mean: float
    min: float
    max: float
    n: int

    @property
    def spread(self) -> float:
        return self.max - self.min

    @classmethod
    def of(cls, values) -> "Stat | None":
        clean = [float(v) for v in values if np.isfinite(v)]
        if not clean:
            return None
        return cls(mean=float(np.mean(clean)), min=min(clean), max=max(clean), n=len(clean))


@dataclass(frozen=True)
class Profile:
    """What a full-range rep measures for one athlete on one camera."""

    source: str
    created: str
    reps: int
    hang_degrees: Stat
    top_degrees: Stat
    clearance: Stat
    version: int = PROFILE_VERSION

    # ── what the rest of the pipeline actually asks it ────────────────────
    @property
    def sweep(self) -> float:
        """Degrees of elbow flexion in a complete pull, bottom to top."""
        return self.hang_degrees.mean - self.top_degrees.mean

    def hang_required(self, cfg) -> float:
        """The elbow angle a hang has to reach to count as full extension."""
        return self.hang_degrees.mean - cfg.ROM_HANG_TOLERANCE_DEGREES

    def clearance_required(self, cfg) -> float:
        """How far the head has to clear the hands, in torso lengths.

        A share of this athlete's reference, not a fixed number.
        """
        return self.clearance.mean * cfg.ROM_CLEARANCE_FRACTION

    def min_flexion(self, cfg) -> float:
        """Degrees of flexion below which a bump in the signal is not a rep."""
        return self.sweep * cfg.REP_MIN_FLEXION_FRACTION

    # ── persistence ───────────────────────────────────────────────────────
    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "created": self.created,
            "reps": self.reps,
            "hang_degrees": asdict(self.hang_degrees),
            "top_degrees": asdict(self.top_degrees),
            "clearance": asdict(self.clearance),
            "derived": {
                "sweep_degrees": round(self.sweep, 1),
            },
        }

    def summary(self) -> dict:
        """One-line-per-row view, for console output."""
        return {
            "source": self.source,
            "measured": f"{self.created} ({self.reps} reps)",
            "hang": f"{self.hang_degrees.mean:.1f}° "
                    f"(spread {self.hang_degrees.spread:.1f}°)",
            "top": f"{self.top_degrees.mean:.1f}° "
                   f"(spread {self.top_degrees.spread:.1f}°)",
            "clearance": f"{self.clearance.mean:.2f} torso lengths "
                         f"(spread {self.clearance.spread:.2f})",
            "sweep": f"{self.sweep:.1f}° of elbow flexion per rep",
        }


def path(cfg) -> Path:
    return Path(cfg.CALIBRATION_PROFILE)


def build(analysis, source: str) -> Profile:
    """Measure a reference clip's reps into a Profile.

    Every rep is used: the clip is the definition. A large `spread` flags a bad reference.
    """
    graded = [r.rom for r in analysis.reps if r.rom is not None]
    hangs = [g.hang_degrees for g in graded]
    tops = [g.peak_degrees for g in graded]
    clears = [g.clearance for g in graded]

    stats = [Stat.of(hangs), Stat.of(tops), Stat.of(clears)]
    if any(s is None for s in stats):
        raise ValueError(
            "The reference clip did not yield usable angles at both ends of a "
            "rep. Check timeline.png for that clip before trusting it."
        )

    return Profile(
        source=source,
        created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        reps=len(analysis.reps),
        hang_degrees=stats[0], top_degrees=stats[1], clearance=stats[2],
    )


def save(profile: Profile, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(profile.as_dict(), indent=2))


def load(src: Path) -> Profile | None:
    """Read a saved profile, or None if it is missing, stale or unreadable.

    Older versions are treated as absent rather than patched; re-run calibrate.py.
    """
    if not src.is_file():
        return None
    try:
        blob = json.loads(src.read_text())
        if blob.get("version") != PROFILE_VERSION:
            return None
        return Profile(
            source=blob["source"], created=blob["created"], reps=blob["reps"],
            hang_degrees=Stat(**blob["hang_degrees"]),
            top_degrees=Stat(**blob["top_degrees"]),
            clearance=Stat(**blob["clearance"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return None
