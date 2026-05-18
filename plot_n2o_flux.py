"""Extract N2O flux per treatment and plot, styled by experimental design.

Dataset-name rules:
- "T[n]"            -> single observation, plot directly.
- "R[r]T[t]C[c]"    -> R=replicate, T=treatment, C=subsample.
                       Average subsamples (C) within each (T, R), then
                       compute mean +- SE across replicates for each T.

Plot styling rules:
- A "project code" maps each treatment to a tuple of categorical variables
  (e.g. NIS-pretest -> (NI, Fertilizer, rate)).
- Each variable is assigned to a visual channel: face color, color
  intensity (light/dark), or hatch pattern. The channel assignment is
  declared per project so the same engine can render any design.

Add a new project by extending PROJECTS at the bottom of this file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE_DEFAULT = Path(__file__).parent / "data.json"
OUT_FILE_DEFAULT = Path(__file__).parent / "n2o_flux.png"

RTC_RE = re.compile(r"^R(\d+)T(\d+)C(\d+)$")
T_RE = re.compile(r"^T(\d+)$")


# ---------------------------------------------------------------------------
# Flux extraction
# ---------------------------------------------------------------------------

def get_n2o_flux(rep_entry: dict) -> float | None:
    for fx in rep_entry.get("footer", {}).get("fluxes", []):
        if fx.get("name", "").lower() == "n2o":
            return float(fx["F_o"])
    return None


def collect(data: dict):
    simple_T: dict[int, list[float]] = defaultdict(list)
    rtc: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ds in data["datasets"]:
        name = next(iter(ds.keys()))
        reps = ds[name].get("reps", {})

        m_t = T_RE.match(name)
        m_rtc = RTC_RE.match(name)

        if m_t:
            t_idx = int(m_t.group(1))
            for rv in reps.values():
                f = get_n2o_flux(rv)
                if f is not None:
                    simple_T[t_idx].append(f)
        elif m_rtc:
            r_idx = int(m_rtc.group(1))
            t_idx = int(m_rtc.group(2))
            for rv in reps.values():
                f = get_n2o_flux(rv)
                if f is not None:
                    rtc[t_idx][r_idx].append(f)
        else:
            print(f"WARN: unrecognized dataset name '{name}', skipped",
                  file=sys.stderr)

    return simple_T, rtc


def reduce_rtc(rtc):
    out = {}
    for t_idx, reps in rtc.items():
        rep_means = [np.mean(v) for v in reps.values() if len(v) > 0]
        n = len(rep_means)
        if n == 0:
            continue
        mean = float(np.mean(rep_means))
        se = float(np.std(rep_means, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[t_idx] = (mean, se, n)
    return out


# ---------------------------------------------------------------------------
# Project / styling definitions
# ---------------------------------------------------------------------------

@dataclass
class StyleChannel:
    """A categorical variable + how to render it on a bar."""
    name: str                       # e.g. "Fertilizer"
    levels: list                    # ordered category list, in legend order
    encoding: str                   # "color" | "intensity" | "hatch"
    palette: dict | None = None     # for color/intensity/hatch lookup
    legend_title: str | None = None


@dataclass
class Project:
    code: str
    treatments: dict                # t_index -> dict of variable->level
    channels: list[StyleChannel]    # one per variable used for styling

    def style_for(self, t_idx: int) -> dict:
        """Return facecolor, hatch, edgecolor for treatment t_idx."""
        meta = self.treatments[t_idx]

        face = "#bdbdbd"
        intensity = "high"   # default to full saturation
        hatch = ""

        # First pass: pick base color from the "color" channel.
        for ch in self.channels:
            level = meta.get(ch.name)
            if ch.encoding == "color":
                face = ch.palette.get(level, face)

        # Second pass: apply intensity (light/dark variants of base color).
        for ch in self.channels:
            level = meta.get(ch.name)
            if ch.encoding == "intensity":
                intensity = ch.palette.get(level, intensity)

        # Third pass: hatch pattern.
        for ch in self.channels:
            level = meta.get(ch.name)
            if ch.encoding == "hatch":
                hatch = ch.palette.get(level, "")

        face = _apply_intensity(face, intensity)
        return {"facecolor": face, "hatch": hatch, "edgecolor": "black"}


def _apply_intensity(hex_color: str, intensity: str) -> str:
    """Lighten the base hex color when intensity == 'low'."""
    if intensity != "low":
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    # Mix 55% toward white.
    mix = 0.55
    r = int(r + (255 - r) * mix)
    g = int(g + (255 - g) * mix)
    b = int(b + (255 - b) * mix)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Project registry
# ---------------------------------------------------------------------------

def _nis_pretest() -> Project:
    NO = "NO"
    treatments = {
        1:  {"NI": NO,     "Fertilizer": NO,                "rate": NO},
        2:  {"NI": NO,     "Fertilizer": "Urea",            "rate": NO},
        3:  {"NI": NO,     "Fertilizer": "Ammonium sulfate","rate": NO},
        4:  {"NI": "DMPP", "Fertilizer": "Urea",            "rate": "low"},
        5:  {"NI": "DMPP", "Fertilizer": "Urea",            "rate": "high"},
        6:  {"NI": "DCD",  "Fertilizer": "Urea",            "rate": "low"},
        7:  {"NI": "DCD",  "Fertilizer": "Urea",            "rate": "high"},
        8:  {"NI": "Neem", "Fertilizer": "Urea",            "rate": "low"},
        9:  {"NI": "Neem", "Fertilizer": "Urea",            "rate": "high"},
        10: {"NI": "DMPP", "Fertilizer": "Ammonium sulfate","rate": "low"},
        11: {"NI": "DMPP", "Fertilizer": "Ammonium sulfate","rate": "high"},
        12: {"NI": "DCD",  "Fertilizer": "Ammonium sulfate","rate": "low"},
        13: {"NI": "DCD",  "Fertilizer": "Ammonium sulfate","rate": "high"},
        14: {"NI": "Neem", "Fertilizer": "Ammonium sulfate","rate": "low"},
        15: {"NI": "Neem", "Fertilizer": "Ammonium sulfate","rate": "high"},
    }
    channels = [
        StyleChannel(
            name="Fertilizer",
            levels=["NO", "Urea", "Ammonium sulfate"],
            encoding="color",
            palette={"NO": "#7a7a7a",
                     "Urea": "#2e7fd1",
                     "Ammonium sulfate": "#d96030"},
            legend_title="N FERTILIZER",
        ),
        StyleChannel(
            name="rate",
            levels=["low", "high"],
            encoding="intensity",
            palette={"low": "low", "high": "high"},
            legend_title="RATE",
        ),
        StyleChannel(
            name="NI",
            levels=["NO", "DMPP", "DCD", "Neem"],
            encoding="hatch",
            palette={"NO": "", "DMPP": "/",
                     "DCD": "x", "Neem": "-"},
            legend_title="INHIBITOR (PATTERN)",
        ),
    ]
    return Project(code="NIS-pretest", treatments=treatments, channels=channels)


PROJECTS: dict[str, Callable[[], Project]] = {
    "NIS-pretest": _nis_pretest,
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _legend_handles(project: Project) -> list:
    handles = []
    for ch in project.channels:
        handles.append(mpatches.Patch(
            color="none", label=ch.legend_title or ch.name))
        for lvl in ch.levels:
            label = "No fertilizer" if (ch.name == "Fertilizer" and lvl == "NO") \
                else ("None (solid)" if (ch.name == "NI" and lvl == "NO") else str(lvl))
            label = f"{label} rate" if ch.name == "rate" else label

            if ch.encoding == "color":
                handles.append(mpatches.Patch(
                    facecolor=ch.palette[lvl], edgecolor="black", label=label))
            elif ch.encoding == "intensity":
                base = "#2e7fd1"  # neutral demo base
                handles.append(mpatches.Patch(
                    facecolor=_apply_intensity(base, ch.palette[lvl]),
                    edgecolor="black", label=label))
            elif ch.encoding == "hatch":
                handles.append(mpatches.Patch(
                    facecolor="white", edgecolor="black",
                    hatch=ch.palette[lvl], label=label))
    return handles


def plot(project: Project, data: dict, out_file: Path):
    simple_T, rtc = collect(data)
    rtc_reduced = reduce_rtc(rtc)

    t_indices = sorted(set(simple_T) | set(rtc))
    if not t_indices:
        print("No treatments found.", file=sys.stderr)
        sys.exit(1)

    labels, means, errs = [], [], []
    for t in t_indices:
        labels.append(f"T{t}")
        if t in rtc_reduced:
            m, se, _ = rtc_reduced[t]
        else:
            vals = simple_T[t]
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) \
                if len(vals) > 1 else 0.0
        means.append(m)
        errs.append(se)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))

    for xi, t, m, se in zip(x, t_indices, means, errs):
        style = project.style_for(t) if t in project.treatments \
            else {"facecolor": "#bdbdbd", "hatch": "", "edgecolor": "black"}
        ax.bar(xi, m, yerr=se, capsize=3, linewidth=0.8, **style)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Treatment")
    ax.set_ylabel(r"N$_2$O flux  ($F_o$, nmol m$^{-2}$ s$^{-1}$)")
    ax.set_title(f"{project.code}  —  {data.get('name','')}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(handles=_legend_handles(project),
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_file}")

    print("\nValues used:")
    print(f"{'Treatment':<10}{'Mean':>12}{'SE':>12}")
    for lab, m, se in zip(labels, means, errs):
        print(f"{lab:<10}{m:>12.5f}{se:>12.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="NIS-pretest",
                    help="Project code (registered in PROJECTS).")
    ap.add_argument("--data", type=Path, default=DATA_FILE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_FILE_DEFAULT)
    args = ap.parse_args()

    if args.project not in PROJECTS:
        print(f"Unknown project '{args.project}'. "
              f"Known: {list(PROJECTS)}", file=sys.stderr)
        sys.exit(2)

    project = PROJECTS[args.project]()
    with args.data.open() as f:
        data = json.load(f)
    plot(project, data, args.out)


if __name__ == "__main__":
    main()
