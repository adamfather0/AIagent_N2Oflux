"""Export every treatment's timestamp + footer fields to CSV.

Scans every *.json under --data-dir. For each (dataset, REP, flux) it
writes one row containing the measurement timestamp and all footer
fields. Multiple flux species (e.g. n2o, co2) yield separate rows.

By default produces a single combined CSV. Use --per-file to instead
emit one CSV per source JSON, named after the JSON stem.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR_DEFAULT = Path(__file__).parent / "data"
OUT_DIR_DEFAULT = Path(__file__).parent / "out"

FOOTER_TOPLEVEL = ["P_o", "T_o", "W_o"]
FLUX_FIELDS = ["name", "F_o", "F_cv", "t_o", "C_o", "a", "C_x",
               "iter", "sei", "ses", "r2", "slope", "domain", "n"]

COLUMNS = (
    ["source_file", "dataset", "treatment", "rep", "timestamp"]
    + FOOTER_TOPLEVEL
    + [f"flux_{k}" for k in FLUX_FIELDS]
)


def parse_timestamp(header: dict) -> str:
    date_str = header.get("Date")
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") \
                .isoformat(sep=" ")
        except ValueError:
            pass
    ts = header.get("gps_time")
    if ts is not None:
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) \
            .isoformat(sep=" ")
    return ""


def rows_from(json_path: Path):
    with json_path.open() as f:
        data = json.load(f)
    for ds in data.get("datasets", []):
        dataset_name = next(iter(ds.keys()))
        treatment = dataset_name[1:] if dataset_name.startswith("T") \
            else dataset_name
        for rep_name, rep in ds[dataset_name].get("reps", {}).items():
            footer = rep.get("footer", {})
            base = {
                "source_file": json_path.name,
                "dataset": dataset_name,
                "treatment": treatment,
                "rep": rep_name,
                "timestamp": parse_timestamp(rep.get("header", {})),
                **{k: footer.get(k) for k in FOOTER_TOPLEVEL},
            }
            fluxes = footer.get("fluxes", [])
            if not fluxes:
                yield {**base,
                       **{f"flux_{k}": None for k in FLUX_FIELDS}}
                continue
            for fx in fluxes:
                yield {**base,
                       **{f"flux_{k}": fx.get(k) for k in FLUX_FIELDS}}


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        n = 0
        for r in rows:
            w.writerow(r)
            n += 1
    print(f"Wrote {out_path}  ({n} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR_DEFAULT,
                    help="Input JSON file, or directory of JSON files.")
    ap.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT,
                    help="Output directory (created if missing).")
    ap.add_argument("--combined-name", default="treatments_footer.csv",
                    help="Filename for the combined CSV (default mode).")
    ap.add_argument("--per", action="store_true",
                    help="Write one CSV per source JSON instead of combined.")
    args = ap.parse_args()

    if args.data.is_file():
        files = [args.data]
    elif args.data.is_dir():
        files = sorted(args.data.glob("*.json"))
    else:
        print(f"--data path does not exist: {args.data}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"No *.json under {args.data}", file=sys.stderr)
        sys.exit(1)

    if args.per or args.data.is_file():
        for fp in files:
            write_csv(rows_from(fp), args.out / f"{fp.stem}.csv")
    else:
        def all_rows():
            for fp in files:
                yield from rows_from(fp)
        write_csv(all_rows(), args.out / args.combined_name)


if __name__ == "__main__":
    main()
