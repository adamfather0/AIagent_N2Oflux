"""Extract N2O flux from each treatment in the dataset and plot.

Naming rule:
- "T[n]"            -> single observation, plot directly.
- "R[r]T[t]C[c]"    -> R=replicate, T=treatment, C=subsample.
                       Average subsamples (C) within each (T, R), then
                       compute mean +- SE across replicates for each T.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = Path(__file__).parent / "data.json"
OUT_FILE = Path(__file__).parent / "n2o_flux.png"

RTC_RE = re.compile(r"^R(\d+)T(\d+)C(\d+)$")
T_RE = re.compile(r"^T(\d+)$")


def get_n2o_flux(rep_entry: dict) -> float | None:
    for fx in rep_entry.get("footer", {}).get("fluxes", []):
        if fx.get("name", "").lower() == "n2o":
            return float(fx["F_o"])
    return None


def collect(data: dict):
    """Returns (simple_T, rtc) where:
       simple_T: list of (t_index, [flux values across reps])
       rtc:      dict t_index -> dict rep -> list of subsample fluxes
    """
    simple_T: dict[int, list[float]] = defaultdict(list)
    rtc: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ds in data["datasets"]:
        name = next(iter(ds.keys()))
        reps = ds[name].get("reps", {})

        m_t = T_RE.match(name)
        m_rtc = RTC_RE.match(name)

        if m_t:
            t_idx = int(m_t.group(1))
            for rk, rv in reps.items():
                f = get_n2o_flux(rv)
                if f is not None:
                    simple_T[t_idx].append(f)
        elif m_rtc:
            r_idx = int(m_rtc.group(1))
            t_idx = int(m_rtc.group(2))
            # Subsample id is encoded in the dataset name; replicates may be
            # split across multiple datasets or stored as REP_x inside one.
            for rk, rv in reps.items():
                f = get_n2o_flux(rv)
                if f is not None:
                    rtc[t_idx][r_idx].append(f)
        else:
            print(f"WARN: unrecognized dataset name '{name}', skipped",
                  file=sys.stderr)

    return simple_T, rtc


def reduce_rtc(rtc: dict[int, dict[int, list[float]]]):
    """For each treatment: average subsamples within rep, then mean +- SE across reps."""
    out: dict[int, tuple[float, float, int]] = {}
    for t_idx, reps in rtc.items():
        rep_means = [np.mean(vals) for vals in reps.values() if len(vals) > 0]
        n = len(rep_means)
        if n == 0:
            continue
        mean = float(np.mean(rep_means))
        se = float(np.std(rep_means, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[t_idx] = (mean, se, n)
    return out


def main():
    with DATA_FILE.open() as f:
        data = json.load(f)

    simple_T, rtc = collect(data)

    t_indices = sorted(set(simple_T) | set(rtc))
    if not t_indices:
        print("No treatments found.", file=sys.stderr)
        sys.exit(1)

    labels, means, errs = [], [], []
    rtc_reduced = reduce_rtc(rtc)

    for t in t_indices:
        labels.append(f"T{t}")
        if t in rtc_reduced:
            m, se, _ = rtc_reduced[t]
            means.append(m)
            errs.append(se)
        else:
            vals = simple_T[t]
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                       if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    colors = ["#4a7fb8" if m >= 0 else "#888888" for m in means]
    bars = ax.bar(x, means, yerr=errs, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Treatment")
    ax.set_ylabel(r"N$_2$O flux  ($F_o$ from footer, nmol m$^{-2}$ s$^{-1}$)")
    ax.set_title(f"N$_2$O flux per treatment  ({data.get('name','')})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for xi, m, se in zip(x, means, errs):
        y = m + (se if m >= 0 else -se)
        va = "bottom" if m >= 0 else "top"
        ax.annotate(f"{m:.3f}", (xi, y),
                    ha="center", va=va, fontsize=8,
                    xytext=(0, 3 if m >= 0 else -3),
                    textcoords="offset points")

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"Wrote {OUT_FILE}")

    print("\nValues used:")
    print(f"{'Treatment':<10}{'Mean':>12}{'SE':>12}")
    for lab, m, se in zip(labels, means, errs):
        print(f"{lab:<10}{m:>12.5f}{se:>12.5f}")


if __name__ == "__main__":
    main()
