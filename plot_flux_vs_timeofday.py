"""Scatter-plot N2O flux vs time-of-day across all JSON files in a folder.

Reads every *.json in --data-dir, extracts (timestamp, flux) for every REP
in every treatment, and plots flux against the hour-of-day. Treatments are
not differentiated; points are coloured by calendar date so multiple days
remain distinguishable.

Timestamp source: header.Date (string, "YYYY-MM-DD HH:MM:SS") falling back
to header.gps_time (unix seconds). Flux source: footer.fluxes[name=n2o].F_o.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR_DEFAULT = Path(__file__).parent / "data"
OUT_FILE_DEFAULT = Path(__file__).parent / "out" / "flux_vs_timeofday.png"


def parse_timestamp(header: dict) -> datetime | None:
    date_str = header.get("Date")
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    ts = header.get("gps_time")
    if ts is not None:
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
    return None


def get_n2o_flux(rep_entry: dict) -> float | None:
    for fx in rep_entry.get("footer", {}).get("fluxes", []):
        if fx.get("name", "").lower() == "n2o":
            return float(fx["F_o"])
    return None


def extract_points(json_path: Path):
    with json_path.open() as f:
        data = json.load(f)
    points = []  # list of (datetime, flux)
    for ds in data.get("datasets", []):
        name = next(iter(ds.keys()))
        reps = ds[name].get("reps", {})
        for rv in reps.values():
            ts = parse_timestamp(rv.get("header", {}))
            flux = get_n2o_flux(rv)
            if ts is None or flux is None:
                continue
            points.append((ts, flux))
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT,
                    help="Folder containing input JSON files.")
    ap.add_argument("--out", type=Path, default=OUT_FILE_DEFAULT)
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("*.json"))
    if not files:
        print(f"No *.json under {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    all_points: list[tuple[datetime, float, str]] = []
    for fp in files:
        pts = extract_points(fp)
        for ts, fx in pts:
            all_points.append((ts, fx, fp.stem))
        print(f"{fp.name}: {len(pts)} points")

    if not all_points:
        print("No (timestamp, flux) pairs found.", file=sys.stderr)
        sys.exit(1)

    # Hour-of-day as fractional hours (HH + MM/60 + SS/3600).
    def hod(t: datetime) -> float:
        return t.hour + t.minute / 60.0 + t.second / 3600.0

    dates = sorted({ts.date() for ts, _, _ in all_points})
    cmap = plt.get_cmap("tab10")
    date_color = {d: cmap(i % 10) for i, d in enumerate(dates)}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for d in dates:
        xs = [hod(ts) for ts, _, _ in all_points if ts.date() == d]
        ys = [fx for ts, fx, _ in all_points if ts.date() == d]
        ax.scatter(xs, ys, s=40, alpha=0.8,
                   color=date_color[d], edgecolor="black", linewidth=0.4,
                   label=d.isoformat())

    ax.set_xlabel("Time of day (hour)")
    ax.set_ylabel(r"N$_2$O flux  ($F_o$, nmol m$^{-2}$ s$^{-1}$)")
    ax.set_title(f"N$_2$O flux vs time-of-day  ({len(all_points)} points, "
                 f"{len(files)} files)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Date", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {args.out}  ({len(all_points)} points across "
          f"{len(dates)} dates)")


if __name__ == "__main__":
    main()
