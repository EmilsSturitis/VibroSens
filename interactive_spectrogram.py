#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

import numpy as np
from scipy import signal

# Avoid permission issues writing Matplotlib cache.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

# Default to TkAgg for interactive GUI if available.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt


MAX_SPECTRO_SAMPLES = 300_000


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


def main():
    parser = argparse.ArgumentParser(
        description="Interactive spectrogram viewer with cursor readout."
    )
    parser.add_argument("input", help="Path to a vibro CSV file.")
    parser.add_argument(
        "--axis",
        choices=["ax", "ay", "az", "mag", "all"],
        default="mag",
        help="Axis to visualize: ax, ay, az, or mag (combined).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SPECTRO_SAMPLES,
        help="Max samples for spectrogram (downsample if larger).",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=2048,
        help="STFT window length in samples.",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=1536,
        help="STFT overlap in samples.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Max frequency to display (Hz).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Path to save the spectrogram image (PNG/PDF).",
    )
    parser.add_argument(
        "--plotly-out",
        default=None,
        help="Write an interactive HTML spectrogram (requires plotly).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Matplotlib backend (e.g., TkAgg, QtAgg).",
    )
    args = parser.parse_args()

    if args.backend:
        try:
            matplotlib.use(args.backend, force=True)
        except Exception as exc:
            raise SystemExit(f"Failed to set backend {args.backend}: {exc}")

    t, ax, ay, az = read_vibro_csv(args.input)
    if args.axis == "ax":
        series = [("aX [g]", ax)]
    elif args.axis == "ay":
        series = [("aY [g]", ay)]
    elif args.axis == "az":
        series = [("aZ [g]", az)]
    elif args.axis == "all":
        series = [("aX [g]", ax), ("aY [g]", ay), ("aZ [g]", az)]
    else:
        series = [("mag", np.sqrt(ax * ax + ay * ay + az * az))]

    if len(t) < 2:
        raise SystemExit("Not enough samples for spectrogram.")
    dt = np.median(np.diff(t))
    fs = 1.0 / dt if dt > 0 else 0.0
    if fs <= 0:
        raise SystemExit("Invalid sampling rate.")

    processed = []
    for label, x in series:
        if len(x) > args.max_samples:
            step = int(np.ceil(len(x) / args.max_samples))
            x = x[::step]
            t_ds = t[::step]
            fs_ds = fs / step
        else:
            t_ds = t
            fs_ds = fs
        nperseg = min(args.nperseg, len(x))
        noverlap = min(args.noverlap, max(0, nperseg - 1))
        f, tt, sxx = signal.spectrogram(
            x,
            fs=fs_ds,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="magnitude",
        )
        sxx_db = 20 * np.log10(sxx + 1e-12)
        if args.fmax is not None:
            f_mask = f <= args.fmax
            f = f[f_mask]
            sxx_db = sxx_db[f_mask, :]
        processed.append((label, f, tt, sxx_db))

    if args.plotly_out:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception as exc:
            raise SystemExit(f"Plotly is required for --plotly-out: {exc}")
        if len(processed) == 1:
            label, f, tt, sxx_db = processed[0]
            fig = go.Figure(
                data=go.Heatmap(
                    x=tt,
                    y=f,
                    z=sxx_db,
                    colorscale="Viridis",
                    colorbar=dict(title="Mag [dB]"),
                )
            )
            fig.update_layout(
                title=f"Spectrogram ({label})",
                xaxis_title="Time [s]",
                yaxis_title="Freq [Hz]",
                height=500,
            )
        else:
            fig = make_subplots(
                rows=len(processed),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[p[0] for p in processed],
            )
            for idx, (label, f, tt, sxx_db) in enumerate(processed, start=1):
                fig.add_trace(
                    go.Heatmap(
                        x=tt,
                        y=f,
                        z=sxx_db,
                        colorscale="Viridis",
                        showscale=(idx == 1),
                        colorbar=dict(title="Mag [dB]") if idx == 1 else None,
                    ),
                    row=idx,
                    col=1,
                )
                fig.update_yaxes(title_text="Freq [Hz]", row=idx, col=1)
            fig.update_xaxes(title_text="Time [s]", row=len(processed), col=1)
            fig.update_layout(title="Spectrogram (stacked)")
        fig.write_html(args.plotly_out)
        print(f"Wrote interactive spectrogram to {args.plotly_out}")
        return

    fig, axes = plt.subplots(
        len(processed), 1, figsize=(12, 5 * len(processed)), sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for axp, (label, f, tt, sxx_db) in zip(axes, processed):
        pcm = axp.pcolormesh(tt, f, sxx_db, shading="gouraud")
        axp.set_ylabel("Freq [Hz]")
        axp.set_title(f"Spectrogram ({label})")
        fig.colorbar(pcm, ax=axp, pad=0.01, label="Mag [dB]")

    axes[-1].set_xlabel("Time [s]")

    readout = axes[0].text(
        0.01,
        0.99,
        "",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        color="w",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )

    def on_move(event):
        if event.inaxes not in axes:
            return
        t_val = event.xdata
        f_val = event.ydata
        if t_val is None or f_val is None:
            return
        readout.set_text(f"t={t_val:.2f}s  f={f_val:.2f}Hz")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"Saved spectrogram to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
