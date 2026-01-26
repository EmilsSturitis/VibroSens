#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Optional
from datetime import datetime

import numpy as np
from scipy import signal

# Avoid permission issues writing Matplotlib cache.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

# Use a non-interactive backend for headless runs.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = "analysis_outputs"
MIN_GAP_SECONDS = 0.5  # Minimum time gap to split flights.
GAP_MULTIPLIER = 5.0  # Gaps larger than this * median dt start a new segment.
TOP_PEAKS = 5
PEAK_MIN_HZ = 2.0
MAX_SPECTRO_SAMPLES = 400_000
MAX_WELCH_SAMPLES = 1_000_000
ENERGY_WINDOW_SEC = 0.5
ENERGY_K = 6.0
MIN_SEGMENT_SEC = 5.0
MIN_GAP_SEC = 2.0


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


def split_by_gaps(t: np.ndarray):
    if len(t) < 2:
        return [slice(0, len(t))], None, None
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    gap_threshold = max(MIN_GAP_SECONDS, GAP_MULTIPLIER * median_dt)
    gap_idx = np.where(dt > gap_threshold)[0]
    segments = []
    start = 0
    for idx in gap_idx:
        end = idx + 1
        segments.append(slice(start, end))
        start = end
    segments.append(slice(start, len(t)))
    return segments, median_dt, gap_threshold


def split_by_energy(
    t: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    energy_k: float,
    min_segment_sec: float,
    min_gap_sec: float,
    energy_window_sec: float,
):
    if len(t) < 2:
        return [slice(0, len(t))], None
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    fs = 1.0 / median_dt if median_dt > 0 else 0.0
    if fs <= 0:
        return [slice(0, len(t))], None
    win = max(1, int(round(energy_window_sec * fs)))
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(mag * mag, kernel, mode="same"))
    med = np.median(rms)
    mad = np.median(np.abs(rms - med))
    thresh = med + energy_k * mad
    active = rms >= thresh

    segments = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            end = i
            segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, len(active)))

    min_len = int(round(min_segment_sec * fs))
    segments = [(s, e) for s, e in segments if (e - s) >= min_len]
    if not segments:
        return [slice(0, len(t))], thresh

    merged = []
    gap_min = int(round(min_gap_sec * fs))
    cur_s, cur_e = segments[0]
    for s, e in segments[1:]:
        if s - cur_e <= gap_min:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    slices = [slice(s, e) for s, e in merged]
    return slices, (thresh, med, mad)


def _welch_spectrum(x: np.ndarray, fs: float):
    x = signal.detrend(x, type="constant")
    if len(x) > MAX_WELCH_SAMPLES:
        step = int(np.ceil(len(x) / MAX_WELCH_SAMPLES))
        x = x[::step]
        fs = fs / step
    nperseg = max(1024, min(16384, int(fs * 2)))
    freqs, psd = signal.welch(
        x, fs=fs, window="hann", nperseg=nperseg, detrend="constant"
    )
    mag = np.sqrt(psd)
    return freqs, mag


def _top_peaks(freqs: np.ndarray, mag: np.ndarray, k: int):
    mask = freqs >= PEAK_MIN_HZ
    freqs = freqs[mask]
    mag = mag[mask]
    if len(freqs) == 0:
        return []
    idx = np.argsort(mag)[-k:][::-1]
    return [(float(freqs[i]), float(mag[i])) for i in idx]


def _apply_clip(ax, ay, az, clip_g):
    if clip_g is None or clip_g <= 0:
        return ax, ay, az
    ax = np.clip(ax, -clip_g, clip_g)
    ay = np.clip(ay, -clip_g, clip_g)
    az = np.clip(az, -clip_g, clip_g)
    return ax, ay, az


def analyze_file(
    path: str,
    output_dir: str,
    segment_mode: str,
    energy_k: float,
    min_segment_sec: float,
    min_gap_sec: float,
    energy_window_sec: float,
    segment_start: "Optional[float]",
    segment_end: "Optional[float]",
    clip_g: "Optional[float]" = None,
):
    t, ax, ay, az = read_vibro_csv(path)
    ax, ay, az = _apply_clip(ax, ay, az, clip_g)
    segments = []
    median_dt = None
    gap_threshold = None
    energy_stats = None
    if segment_mode == "gap":
        segments, median_dt, gap_threshold = split_by_gaps(t)
    elif segment_mode == "energy":
        segments, energy_stats = split_by_energy(
            t,
            ax,
            ay,
            az,
            energy_k,
            min_segment_sec,
            min_gap_sec,
            energy_window_sec,
        )
        median_dt = float(np.median(np.diff(t))) if len(t) > 1 else None
    elif segment_mode == "manual":
        if segment_start is None or segment_end is None:
            raise SystemExit(
                "--segment-start and --segment-end are required for manual mode."
            )
        t_min = float(t[0])
        t_max = float(t[-1])
        start_sec = max(segment_start, t_min)
        end_sec = min(segment_end, t_max)
        if end_sec <= start_sec:
            raise SystemExit("Manual segment range is empty after clamping to data.")
        start_idx = int(np.searchsorted(t, start_sec, side="left"))
        end_idx = int(np.searchsorted(t, end_sec, side="right"))
        segments = [slice(start_idx, end_idx)]
        median_dt = float(np.median(np.diff(t))) if len(t) > 1 else None
    else:
        raise ValueError(f"Unknown segment mode: {segment_mode}")

    base = os.path.splitext(os.path.basename(path))[0]
    summary_lines = []
    summary_lines.append(f"File: {path}")
    summary_lines.append(f"Total samples: {len(t)}")
    if clip_g is not None and clip_g > 0:
        summary_lines.append(f"G clip (abs): {clip_g:.6g} g")
    if median_dt is not None:
        summary_lines.append(f"Median dt: {median_dt:.6f}s")
    if gap_threshold is not None:
        summary_lines.append(f"Gap threshold: {gap_threshold:.3f}s")
    if energy_stats is not None:
        thresh, med, mad = energy_stats
        summary_lines.append(
            f"Energy threshold (median + {energy_k:.2f}*MAD): {thresh:.6f} (med {med:.6f}, MAD {mad:.6f})"
        )
    if segment_mode == "manual":
        summary_lines.append(
            f"Manual segment requested: {segment_start:.3f}s to {segment_end:.3f}s"
        )
    summary_lines.append(f"Segments: {len(segments)}")

    if len(t) > 3:
        seg_dt = np.median(np.diff(t))
        fs = 1.0 / seg_dt if seg_dt > 0 else 0.0
        if fs > 0:
            fig, ax_main = plt.subplots(1, 1, figsize=(12, 5))
            mag = np.sqrt(ax * ax + ay * ay + az * az)
            nperseg = max(256, min(4096, int(fs * 2)))
            noverlap = int(nperseg * 0.75)
            f, tt, sxx = signal.spectrogram(
                mag,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                mode="magnitude",
            )
            sxx_db = 20 * np.log10(sxx + 1e-12)
            pcm = ax_main.pcolormesh(tt, f, sxx_db, shading="gouraud")
            ax_main.set_ylabel("Freq [Hz]")
            ax_main.set_xlabel("Time [s]")
            ax_main.set_title(f"{base} - Combined Spectrogram")
            fig.colorbar(pcm, ax=ax_main, pad=0.01, label="Mag [dB]")
            for idx, seg in enumerate(segments, start=1):
                t_start = t[seg.start]
                t_end = t[seg.stop - 1] if seg.stop > seg.start else t[seg.start]
                ax_main.axvline(t_start, color="w", lw=1.0)
                ax_main.axvline(t_end, color="w", lw=1.0)
                ax_main.text(
                    t_start,
                    f.max() * 0.98,
                    f"S{idx}",
                    color="w",
                    fontsize=8,
                    va="top",
                )
            fig.tight_layout()
            main_path = os.path.join(output_dir, f"{base}_combined_spectrogram.png")
            fig.savefig(main_path, dpi=150)
            plt.close(fig)

    for s_idx, seg in enumerate(segments, start=1):
        ts = t[seg]
        if len(ts) < 4:
            continue
        seg_dt = np.median(np.diff(ts))
        fs = 1.0 / seg_dt if seg_dt > 0 else 0.0
        summary_lines.append(f"  Segment {s_idx}: {len(ts)} samples, fs ~ {fs:.2f} Hz")
        fs_seg = fs

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax_idx, (axis_data, label) in enumerate(
            [(ax[seg], "aX [g]"), (ay[seg], "aY [g]"), (az[seg], "aZ [g]")]
        ):
            freqs, mag = _welch_spectrum(axis_data, fs_seg)
            mag_db = 20 * np.log10(mag + 1e-12)
            axes[ax_idx].plot(freqs, mag_db, lw=1)
            axes[ax_idx].set_ylabel(f"{label}\nmag (dB)")
            axes[ax_idx].grid(True, alpha=0.3)
            peaks = _top_peaks(freqs, mag, TOP_PEAKS)
            for f0, m0 in peaks:
                axes[ax_idx].text(
                    f0,
                    20 * np.log10(m0 + 1e-12),
                    f"{f0:.1f} Hz",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )
                summary_lines.append(
                    f"    Segment {s_idx} {label} peak: {f0:.2f} Hz (mag {m0:.4e})"
                )
        axes[-1].set_xlabel("Frequency [Hz]")
        fig.suptitle(f"{base} - Segment {s_idx} Spectrum")
        fig.tight_layout()
        spectrum_path = os.path.join(output_dir, f"{base}_segment{s_idx}_spectrum.png")
        fig.savefig(spectrum_path, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax_idx, (axis_data, label) in enumerate(
            [(ax[seg], "aX [g]"), (ay[seg], "aY [g]"), (az[seg], "aZ [g]")]
        ):
            fs_spec = fs_seg
            if len(axis_data) > MAX_SPECTRO_SAMPLES:
                step = int(np.ceil(len(axis_data) / MAX_SPECTRO_SAMPLES))
                axis_data = axis_data[::step]
                fs_spec = fs_spec / step
            nperseg = max(256, min(4096, int(fs_spec * 2)))
            noverlap = int(nperseg * 0.75)
            f, tt, sxx = signal.spectrogram(
                axis_data,
                fs=fs_spec,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                mode="magnitude",
            )
            sxx_db = 20 * np.log10(sxx + 1e-12)
            pcm = axes[ax_idx].pcolormesh(tt, f, sxx_db, shading="gouraud")
            axes[ax_idx].set_ylabel(f"{label}\nFreq [Hz]")
            axes[ax_idx].grid(False)
            fig.colorbar(pcm, ax=axes[ax_idx], pad=0.01, label="Mag [dB]")
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"{base} - Segment {s_idx} Spectrogram")
        fig.tight_layout()
        spectro_path = os.path.join(output_dir, f"{base}_segment{s_idx}_spectrogram.png")
        fig.savefig(spectro_path, dpi=150)
        plt.close(fig)

    summary_path = os.path.join(output_dir, f"{base}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze vibration CSV and generate spectra/spectrograms."
    )
    parser.add_argument(
        "input",
        help="Path to a vibro CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write results into (default: analysis_outputs/<file-base>/).",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["gap", "energy", "manual"],
        default="gap",
        help="Segmentation method: gap (time gaps), energy (activity), or manual (time range).",
    )
    parser.add_argument(
        "--segment-start",
        type=float,
        default=None,
        help="Manual segment start time in seconds (requires --segment-mode manual).",
    )
    parser.add_argument(
        "--segment-end",
        type=float,
        default=None,
        help="Manual segment end time in seconds (requires --segment-mode manual).",
    )
    parser.add_argument(
        "--energy-k",
        type=float,
        default=ENERGY_K,
        help="MAD multiplier for adaptive energy thresholding.",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=MIN_SEGMENT_SEC,
        help="Minimum segment duration for energy segmentation.",
    )
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=MIN_GAP_SEC,
        help="Minimum gap to split segments for energy segmentation.",
    )
    parser.add_argument(
        "--energy-window-sec",
        type=float,
        default=ENERGY_WINDOW_SEC,
        help="Window size for RMS energy smoothing (seconds).",
    )
    parser.add_argument(
        "--clip-g",
        type=float,
        default=None,
        help="Cap acceleration values to +/- this threshold in g.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    base = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, base)
    os.makedirs(output_dir, exist_ok=True)
    analyze_file(
        args.input,
        output_dir,
        args.segment_mode,
        args.energy_k,
        args.min_segment_sec,
        args.min_gap_sec,
        args.energy_window_sec,
        args.segment_start,
        args.segment_end,
        args.clip_g,
    )
    print(f"Done. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
