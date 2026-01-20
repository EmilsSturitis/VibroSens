#!/usr/bin/env python3
import csv
import os
import sys
from datetime import datetime

import numpy as np


def _parse_datetime(date_str: str, time_str: str) -> datetime:
    day, month, year = [int(x) for x in date_str.split(".")]
    t_parts = time_str.split(":")
    if len(t_parts) != 4:
        raise ValueError(f"Unexpected time format: {time_str}")
    hour, minute, second = [int(x) for x in t_parts[:3]]
    frac_raw = t_parts[3]
    frac = int(frac_raw) if frac_raw else 0
    if len(frac_raw) == 4:
        microsecond = frac * 100
    elif len(frac_raw) == 3:
        microsecond = frac * 1000
    else:
        microsecond = frac * 1000
    return datetime(year, month, day, hour, minute, second, microsecond)


def read_vibro_csv(path: str):
    header = None
    rows = []
    with open(path, "r", newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            line = line.lstrip("\ufeff")
            if line.startswith("@"):
                if line.startswith("@Date;"):
                    header = line[1:].split(";")
                continue
            if header is None:
                continue
            rows.append(line.split(";"))

    if header is None:
        raise ValueError(f"Header not found in {path}")

    col_idx = {name: i for i, name in enumerate(header)}
    required = ["Date", "Time", "aX [g]", "aY [g]", "aZ [g]"]
    for name in required:
        if name not in col_idx:
            raise ValueError(f"Missing column {name} in {path}")

    times = []
    ax = []
    ay = []
    az = []
    for row in rows:
        try:
            dt = _parse_datetime(row[col_idx["Date"]], row[col_idx["Time"]])
        except Exception:
            continue
        times.append(dt)
        ax.append(float(row[col_idx["aX [g]"]]))
        ay.append(float(row[col_idx["aY [g]"]]))
        az.append(float(row[col_idx["aZ [g]"]]))

    if not times:
        raise ValueError(f"No data rows found in {path}")

    t0 = times[0]
    t = np.array([(dt - t0).total_seconds() for dt in times], dtype=float)
    return t, np.array(ax), np.array(ay), np.array(az)


def _collect_inputs(entries):
    files = []
    for entry in entries:
        p = entry["path"]
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(".csv"):
                    files.append(
                        {
                            "path": os.path.join(p, name),
                            "start": entry["start"],
                            "end": entry["end"],
                        }
                    )
        else:
            files.append(entry)
    return files


def _window_slice(t, start, end):
    if start is None and end is None:
        return slice(0, len(t))
    t_min = float(t[0])
    t_max = float(t[-1])
    start_sec = t_min if start is None else max(float(start), t_min)
    end_sec = t_max if end is None else min(float(end), t_max)
    if end_sec <= start_sec:
        raise ValueError("Requested time window is empty after clamping to data.")
    start_idx = int(np.searchsorted(t, start_sec, side="left"))
    end_idx = int(np.searchsorted(t, end_sec, side="right"))
    return slice(start_idx, end_idx)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def quantify_file(path, start, end):
    t, ax, ay, az = read_vibro_csv(path)
    seg = _window_slice(t, start, end)
    t = t[seg]
    ax = ax[seg]
    ay = ay[seg]
    az = az[seg]
    mag = np.sqrt(ax * ax + ay * ay + az * az)

    rms_mag = _rms(mag)
    rms_ax = _rms(ax)
    rms_ay = _rms(ay)
    rms_az = _rms(az)
    peak_mag = float(np.max(np.abs(mag)))
    median_mag = float(np.median(mag))
    p95_mag = float(np.percentile(mag, 95))
    mean_mag = float(np.mean(mag))
    crest = peak_mag / rms_mag if rms_mag > 0 else float("inf")

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return {
        "file": path,
        "samples": int(len(t)),
        "duration_s": duration,
        "rms_mag": rms_mag,
        "rms_ax": rms_ax,
        "rms_ay": rms_ay,
        "rms_az": rms_az,
        "mean_mag": mean_mag,
        "median_mag": median_mag,
        "p95_mag": p95_mag,
        "peak_mag": peak_mag,
        "crest_factor": crest,
    }


def _print_table(rows):
    headers = [
        "file",
        "samples",
        "duration_s",
        "rms_mag",
        "p95_mag",
        "peak_mag",
        "rms_ax",
        "rms_ay",
        "rms_az",
        "crest_factor",
    ]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                text = f"{val:.6g}"
            else:
                text = str(val)
            col_widths[h] = max(col_widths[h], len(text))

    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        parts = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                text = f"{val:.6g}"
            else:
                text = str(val)
            parts.append(text.ljust(col_widths[h]))
        print("  ".join(parts))


def _parse_args(argv):
    entries = []
    sort_by = "rms_mag"
    out_csv = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--sort-by":
            if i + 1 >= len(argv):
                raise SystemExit("--sort-by requires a value.")
            sort_by = argv[i + 1]
            i += 2
            continue
        if tok == "--out-csv":
            if i + 1 >= len(argv):
                raise SystemExit("--out-csv requires a value.")
            out_csv = argv[i + 1]
            i += 2
            continue
        if tok in ("--start", "--end"):
            if not entries:
                raise SystemExit(f"{tok} must follow a file or directory.")
            if i + 1 >= len(argv):
                raise SystemExit(f"{tok} requires a value.")
            val = float(argv[i + 1])
            if tok == "--start":
                entries[-1]["start"] = val
            else:
                entries[-1]["end"] = val
            i += 2
            continue
        if tok.startswith("--"):
            raise SystemExit(f"Unknown option: {tok}")
        entries.append({"path": tok, "start": None, "end": None})
        i += 1
    if not entries:
        raise SystemExit("No input CSV files found.")
    return entries, sort_by, out_csv


def main():
    entries, sort_by, out_csv = _parse_args(sys.argv[1:])

    allowed_sort = {
        "rms_mag",
        "p95_mag",
        "peak_mag",
        "crest_factor",
        "duration_s",
        "samples",
    }
    if sort_by not in allowed_sort:
        raise SystemExit(f"Invalid --sort-by: {sort_by}")

    files = _collect_inputs(entries)
    if not files:
        raise SystemExit("No input CSV files found.")

    rows = []
    for entry in files:
        path = entry["path"]
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        rows.append(quantify_file(path, entry["start"], entry["end"]))

    if not rows:
        raise SystemExit("No valid CSV files to process.")

    rows.sort(key=lambda r: r[sort_by])
    _print_table(rows)

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary to {out_csv}")


if __name__ == "__main__":
    main()
