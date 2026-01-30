#!/usr/bin/env python3
import csv
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import analyze_vibro
import quantify_vibro
import numpy as np
from scipy import signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def _format_metrics(metrics):
    lines = [
        f"File: {metrics['file']}",
        f"Samples: {metrics['samples']}",
        f"Duration [s]: {metrics['duration_s']:.3f}",
        f"RMS mag [g]: {metrics['rms_mag']:.6g}",
        f"P95 mag [g]: {metrics['p95_mag']:.6g}",
        f"Peak mag [g]: {metrics['peak_mag']:.6g}",
        f"RMS aX [g]: {metrics['rms_ax']:.6g}",
        f"RMS aY [g]: {metrics['rms_ay']:.6g}",
        f"RMS aZ [g]: {metrics['rms_az']:.6g}",
        f"Crest factor: {metrics['crest_factor']:.6g}",
    ]
    return "\n".join(lines)


class VibroGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VibroSens Analyzer")
        self.geometry("760x520")
        self.resizable(True, True)

        self.file_var = tk.StringVar()
        self.log_file_var = tk.StringVar()
        self.log_offset_var = tk.StringVar()
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.clip_g_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a CSV to begin.")

        self._build_ui()
        self._preview_cid = None
        self._preview_data = None
        self._max_freq_hz = 250.0

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        file_frame = tk.Frame(self)
        file_frame.pack(fill="x", **pad)
        tk.Label(file_frame, text="CSV file:").pack(side="left")
        tk.Entry(file_frame, textvariable=self.file_var, width=60).pack(
            side="left", padx=6, fill="x", expand=True
        )
        tk.Button(file_frame, text="Browse", command=self._pick_file).pack(side="left")

        log_frame = tk.Frame(self)
        log_frame.pack(fill="x", **pad)
        tk.Label(log_frame, text="ArduPilot log (CSV/BIN, optional):").pack(
            side="left"
        )
        tk.Entry(log_frame, textvariable=self.log_file_var, width=50).pack(
            side="left", padx=6, fill="x", expand=True
        )
        tk.Button(log_frame, text="Browse", command=self._pick_log_file).pack(
            side="left"
        )
        tk.Label(log_frame, text="Offset [s]:").pack(side="left", padx=(8, 0))
        tk.Entry(log_frame, textvariable=self.log_offset_var, width=8).pack(
            side="left", padx=4
        )
        self.log_offset_scale = tk.Scale(
            log_frame,
            from_=-360.0,
            to=360.0,
            resolution=0.1,
            orient="horizontal",
            length=180,
            showvalue=False,
            command=self._on_offset_scale,
        )
        self.log_offset_scale.pack(side="left", padx=4)
        self._offset_trace_id = self.log_offset_var.trace_add(
            "write", self._on_offset_entry
        )

        win_frame = tk.Frame(self)
        win_frame.pack(fill="x", **pad)
        tk.Label(win_frame, text="Segment start [s]:").pack(side="left")
        tk.Entry(win_frame, textvariable=self.start_var, width=10).pack(
            side="left", padx=6
        )
        tk.Label(win_frame, text="Segment end [s]:").pack(side="left")
        tk.Entry(win_frame, textvariable=self.end_var, width=10).pack(
            side="left", padx=6
        )
        tk.Label(win_frame, text="G clip (abs) [g]:").pack(side="left")
        tk.Entry(win_frame, textvariable=self.clip_g_var, width=8).pack(
            side="left", padx=6
        )

        out_frame = tk.Frame(self)
        out_frame.pack(fill="x", **pad)
        tk.Label(out_frame, text="Output dir (optional):").pack(side="left")
        tk.Entry(out_frame, textvariable=self.output_var, width=60).pack(
            side="left", padx=6, fill="x", expand=True
        )
        tk.Button(out_frame, text="Browse", command=self._pick_output).pack(side="left")

        run_frame = tk.Frame(self)
        run_frame.pack(fill="x", **pad)
        tk.Button(run_frame, text="Run analysis", command=self._run).pack(side="left")
        tk.Button(run_frame, text="Preview plot", command=self._preview_plot).pack(
            side="left", padx=6
        )
        tk.Button(run_frame, text="Axis plots", command=self._preview_axis_plots).pack(
            side="left", padx=6
        )
        tk.Button(run_frame, text="PWM plot", command=self._preview_pwm_plot).pack(
            side="left", padx=6
        )
        tk.Label(run_frame, textvariable=self.status_var).pack(side="left", padx=10)

        metrics_frame = tk.Frame(self)
        metrics_frame.pack(fill="both", expand=True, **pad)
        tk.Label(metrics_frame, text="Metrics:").pack(anchor="w")
        self.metrics_text = tk.Text(metrics_frame, height=12)
        self.metrics_text.pack(fill="both", expand=True)

        plot_frame = tk.Frame(self)
        plot_frame.pack(fill="both", expand=True, **pad)
        tk.Label(plot_frame, text="Preview (magnitude vs time):").pack(anchor="w")
        self.preview_fig = Figure(figsize=(6.5, 3.2), dpi=100)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_ax2 = None
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=plot_frame)
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.preview_toolbar = NavigationToolbar2Tk(
            self.preview_canvas, plot_frame, pack_toolbar=False
        )
        self.preview_toolbar.update()
        self.preview_toolbar.pack(fill="x")
        self.preview_readout = tk.Label(plot_frame, text="")
        self.preview_readout.pack(anchor="w")

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select vibro CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)
            self._preview_plot()

    def _pick_output(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_var.set(path)

    def _pick_log_file(self):
        path = filedialog.askopenfilename(
            title="Select ArduPilot log",
            filetypes=[
                ("ArduPilot logs", "*.csv;*.bin"),
                ("CSV files", "*.csv"),
                ("BIN files", "*.bin"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.log_file_var.set(path)
            self._preview_plot()

    def _run(self):
        path = self.file_var.get().strip()
        start_raw = self.start_var.get().strip()
        end_raw = self.end_var.get().strip()
        output_dir = self.output_var.get().strip()

        if not path:
            messagebox.showerror("Missing file", "Please choose a CSV file.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Missing file", f"File not found:\n{path}")
            return
        if not start_raw or not end_raw:
            messagebox.showerror(
                "Missing window", "Please enter both start and end times."
            )
            return
        try:
            start = float(start_raw)
            end = float(end_raw)
        except ValueError:
            messagebox.showerror("Invalid window", "Start and end must be numbers.")
            return
        try:
            clip_g = self._parse_clip_g()
        except ValueError as exc:
            messagebox.showerror("Invalid G clip", str(exc))
            return
        if end <= start:
            messagebox.showerror("Invalid window", "End must be greater than start.")
            return

        base = os.path.splitext(os.path.basename(path))[0]
        if not output_dir:
            output_dir = os.path.join(analyze_vibro.DEFAULT_OUTPUT_DIR, base)
        os.makedirs(output_dir, exist_ok=True)

        self.status_var.set("Running analysis...")
        self.update_idletasks()
        try:
            analyze_vibro.analyze_file(
                path,
                output_dir,
                "manual",
                analyze_vibro.ENERGY_K,
                analyze_vibro.MIN_SEGMENT_SEC,
                analyze_vibro.MIN_GAP_SEC,
                analyze_vibro.ENERGY_WINDOW_SEC,
                start,
                end,
                clip_g,
            )
            metrics = quantify_vibro.quantify_file(path, start, end, clip_g)
            metrics_text = _format_metrics(metrics)
            if clip_g:
                metrics_text += f"\nG clip (abs) [g]: {clip_g:.6g}"
            self.metrics_text.delete("1.0", tk.END)
            self.metrics_text.insert(tk.END, metrics_text)
            metrics_path = os.path.join(output_dir, f"{base}_metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(metrics_text + "\n")
            self.status_var.set(f"Done. Outputs in {output_dir}")
        except Exception as exc:
            self.status_var.set("Failed.")
            messagebox.showerror("Error", str(exc))

    def _preview_plot(self):
        path = self.file_var.get().strip()
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showerror("Missing file", f"File not found:\n{path}")
            return
        try:
            t, ax, ay, az = quantify_vibro.read_vibro_csv(path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        try:
            clip_g = self._parse_clip_g()
        except ValueError as exc:
            messagebox.showerror("Invalid G clip", str(exc))
            return
        ax, ay, az = self._apply_clip(ax, ay, az, clip_g)
        mag = (ax * ax + ay * ay + az * az) ** 0.5
        self.preview_fig.clear()
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_ax2 = None
        label = "Mag [g]"
        if clip_g:
            label = f"Mag [g] (clip {clip_g:g})"
        self.preview_ax.plot(t, mag, lw=0.8, color="tab:blue", label=label)
        self.preview_ax.set_xlabel("Time [s]")
        self.preview_ax.set_ylabel("Magnitude [g]")
        self.preview_ax.set_title(os.path.basename(path))
        self.preview_ax.grid(True, alpha=0.3)

        start_raw = self.start_var.get().strip()
        end_raw = self.end_var.get().strip()
        try:
            if start_raw:
                self.preview_ax.axvline(float(start_raw), color="r", lw=1)
            if end_raw:
                self.preview_ax.axvline(float(end_raw), color="r", lw=1)
        except ValueError:
            pass
        self._overlay_log(self.preview_ax, t)
        log_path = self.log_file_var.get().strip()
        if log_path:
            offset = self._parse_log_offset()
            if offset:
                self.preview_ax.set_title(
                    f"{os.path.basename(path)} (log offset {offset:+.3f}s)"
                )
        self._preview_data = (t, mag)
        if self._preview_cid is None:
            self._preview_cid = self.preview_canvas.mpl_connect(
                "motion_notify_event", self._on_preview_hover
            )
        self.preview_canvas.draw_idle()

    def _on_preview_hover(self, event):
        if self._preview_data is None:
            return
        if event.inaxes != self.preview_ax or event.xdata is None:
            self.preview_readout.config(text="")
            return
        t, mag = self._preview_data
        idx = int(np.clip(np.searchsorted(t, event.xdata), 0, len(t) - 1))
        self.preview_readout.config(
            text=f"t={t[idx]:.3f}s  mag={mag[idx]:.6g} g"
        )

    def _window_slice(self, t, start_raw, end_raw):
        if not start_raw and not end_raw:
            return slice(0, len(t))
        t_min = float(t[0])
        t_max = float(t[-1])
        start_sec = t_min if not start_raw else max(float(start_raw), t_min)
        end_sec = t_max if not end_raw else min(float(end_raw), t_max)
        if end_sec <= start_sec:
            raise ValueError("Requested time window is empty after clamping to data.")
        start_idx = int(np.searchsorted(t, start_sec, side="left"))
        end_idx = int(np.searchsorted(t, end_sec, side="right"))
        return slice(start_idx, end_idx)

    def _parse_log_offset(self):
        raw = self.log_offset_var.get().strip()
        if not raw:
            return 0.0
        try:
            return float(raw)
        except ValueError:
            return 0.0

    def _on_offset_scale(self, val):
        try:
            offset = float(val)
        except ValueError:
            return
        self.log_offset_var.set(f"{offset:.3f}")
        self._preview_plot()

    def _on_offset_entry(self, *_args):
        try:
            offset = float(self.log_offset_var.get().strip())
        except ValueError:
            return
        self.log_offset_scale.set(offset)

    def _parse_clip_g(self):
        raw = self.clip_g_var.get().strip()
        if not raw:
            return None
        try:
            val = float(raw)
        except ValueError:
            raise ValueError("G clip must be a number.")
        if val < 0:
            raise ValueError("G clip must be >= 0.")
        return val

    def _apply_clip(self, ax, ay, az, clip_g):
        if clip_g is None or clip_g <= 0:
            return ax, ay, az
        ax = np.clip(ax, -clip_g, clip_g)
        ay = np.clip(ay, -clip_g, clip_g)
        az = np.clip(az, -clip_g, clip_g)
        return ax, ay, az

    def _read_ardupilot_csv(self, path):
        with open(path, "r", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if not reader.fieldnames:
                raise ValueError("No header found in ArduPilot CSV.")
            headers = reader.fieldnames

            time_col = None
            for name in ("TimeUS", "TimeMS", "TimeS", "Time", "time"):
                if name in headers:
                    time_col = name
                    break
            if time_col is None:
                raise ValueError("No time column found (TimeUS/TimeMS/Time/TimeS).")

            pwm_cols = []
            rpm_cols = []
            for name in headers:
                lower = name.lower()
                if "rpm" in lower:
                    rpm_cols.append(name)
                elif (
                    "pwm" in lower
                    or "rcou" in lower
                    or "servo" in lower
                    or re.search(r"\bmot\d+\b", lower)
                ):
                    pwm_cols.append(name)

            cols = rpm_cols + pwm_cols
            if not cols:
                raise ValueError("No PWM/RPM columns found in ArduPilot CSV.")

            time_vals = []
            series = {name: [] for name in cols}
            for row in reader:
                raw_t = row.get(time_col, "").strip()
                if not raw_t:
                    continue
                try:
                    t_val = float(raw_t)
                except ValueError:
                    continue
                for name in cols:
                    raw = row.get(name, "").strip()
                    try:
                        series[name].append(float(raw))
                    except ValueError:
                        series[name].append(float("nan"))
                time_vals.append(t_val)

        if not time_vals:
            raise ValueError("No time samples parsed from ArduPilot CSV.")

        time_vals = np.array(time_vals, dtype=float)
        if time_col.lower().endswith("us"):
            time_vals = time_vals * 1e-6
        elif time_col.lower().endswith("ms"):
            time_vals = time_vals * 1e-3
        else:
            if np.nanmax(time_vals) > 1e5:
                time_vals = time_vals * 1e-6
            elif np.nanmax(time_vals) > 1e3:
                time_vals = time_vals * 1e-3
        time_vals = time_vals - time_vals[0]

        series_np = {name: np.array(vals, dtype=float) for name, vals in series.items()}
        return time_vals, series_np

    def _read_ardupilot_bin(self, path):
        try:
            from pymavlink import mavutil
        except Exception as exc:
            raise ValueError(
                "pymavlink is required to read .bin logs. "
                "Install with: pip install pymavlink"
            ) from exc

        mlog = mavutil.mavlink_connection(path)
        time_vals = []
        series = {}
        time_fields = {
            "timeus",
            "time_us",
            "time_usec",
            "timems",
            "time_ms",
            "time_msec",
            "times",
            "time_s",
            "time",
            "timeusec",
        }
        pwm_msg_types = {"RCOU", "SERVO_OUTPUT_RAW", "RCIN"}
        rpm_msg_types = {"RPM", "ESC", "ESC2", "ESC3", "ESC4", "ESC5", "ESC6", "ESC7", "ESC8"}

        def _to_float(val):
            try:
                return float(val)
            except Exception:
                return None

        def _extract_time(d, msg):
            candidates = [
                ("TimeUS", 1e-6),
                ("time_us", 1e-6),
                ("time_usec", 1e-6),
                ("TimeMS", 1e-3),
                ("time_ms", 1e-3),
                ("time_msec", 1e-3),
                ("TimeS", 1.0),
                ("time_s", 1.0),
                ("Time", 1.0),
                ("time", 1.0),
            ]
            for key, scale in candidates:
                if key in d:
                    val = _to_float(d.get(key))
                    if val is None:
                        continue
                    if key.lower() in ("time", "times") and val > 1e6:
                        return val * 1e-6
                    if key.lower() in ("time", "times") and val > 1e3:
                        return val * 1e-3
                    return val * scale
            ts = getattr(msg, "_timestamp", None)
            val = _to_float(ts)
            if val is None:
                return None
            return val

        def _is_pwm_field(msg_type, field):
            lower = field.lower()
            if lower in time_fields or lower == "index":
                return False
            if "pwm" in lower:
                return True
            if msg_type in pwm_msg_types:
                if "servo" in lower:
                    return True
                if re.fullmatch(r"c\d+", lower):
                    return True
                if re.fullmatch(r"chan\d+_raw", lower):
                    return True
                if re.fullmatch(r"servo\d+_raw", lower):
                    return True
                if re.fullmatch(r"servo\d+", lower):
                    return True
            return False

        def _is_rpm_field(msg_type, field):
            lower = field.lower()
            if lower in time_fields or lower == "index":
                return False
            if "rpm" in lower:
                return True
            if msg_type in rpm_msg_types and "rpm" in lower:
                return True
            return False

        while True:
            msg = mlog.recv_match(blocking=False)
            if msg is None:
                break
            d = msg.to_dict()
            msg_type = d.get("mavpackettype", msg.get_type())
            t_val = _extract_time(d, msg)
            if t_val is None:
                continue

            fields = {}
            for key, val in d.items():
                if key == "mavpackettype":
                    continue
                fval = _to_float(val)
                if fval is None:
                    continue
                if _is_rpm_field(msg_type, key):
                    fields[f"{msg_type}.{key}"] = fval
                elif _is_pwm_field(msg_type, key):
                    fields[f"{msg_type}.{key}"] = fval

            if not fields:
                continue

            for name in fields:
                if name not in series:
                    series[name] = [float("nan")] * len(time_vals)
            for name, vals in series.items():
                vals.append(fields.get(name, float("nan")))
            time_vals.append(t_val)

        if not time_vals:
            raise ValueError("No time samples parsed from ArduPilot BIN.")

        time_vals = np.array(time_vals, dtype=float)
        time_vals = time_vals - time_vals[0]
        series_np = {name: np.array(vals, dtype=float) for name, vals in series.items()}
        return time_vals, series_np

    def _read_ardupilot_log(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bin":
            return self._read_ardupilot_bin(path)
        return self._read_ardupilot_csv(path)

    def _overlay_log(self, ax_main, t_main):
        log_path = self.log_file_var.get().strip()
        if not log_path:
            return
        if not os.path.exists(log_path):
            messagebox.showerror("Missing log", f"Log file not found:\n{log_path}")
            return
        try:
            log_t, series = self._read_ardupilot_log(log_path)
        except Exception as exc:
            messagebox.showerror("Log error", str(exc))
            return

        offset = self._parse_log_offset()
        log_t = log_t + offset
        self.preview_ax2 = ax_main.twinx()
        colors = [
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
        ]
        labels = []
        has_rpm = False
        t_min = float(t_main[0]) if len(t_main) else 0.0
        t_max = float(t_main[-1]) if len(t_main) else 0.0
        mask = (log_t >= t_min) & (log_t <= t_max)
        prefer = ["RCOU.C1", "RCOU.C2", "RCOU.C3", "RCOU.C4"]
        if any(name in series for name in prefer):
            ordered = [(name, series[name]) for name in prefer if name in series]
        else:
            ordered = list(series.items())

        active_baseline = 1100.0
        active_tol = 5.0
        if ordered:
            stacked = np.vstack([vals for _, vals in ordered])
            finite_any = np.isfinite(stacked).any(axis=0)
            active = np.any(np.abs(stacked - active_baseline) > active_tol, axis=0)
            active = active & finite_any
            if active.any():
                start_idx = int(np.argmax(active))
                log_t = log_t[start_idx:]
                for name in list(series.keys()):
                    series[name] = series[name][start_idx:]
                ordered = [(name, series[name]) for name, _ in ordered]
                mask = (log_t >= t_min) & (log_t <= t_max)

        if not mask.any() and len(log_t):
            x_min = min(t_min, float(log_t[0]))
            x_max = max(t_max, float(log_t[-1]))
            ax_main.set_xlim(x_min, x_max)

        for idx, (name, values) in enumerate(ordered):
            if idx >= len(colors):
                break
            if mask.any():
                values = values[mask]
                log_plot_t = log_t[mask]
            else:
                log_plot_t = log_t
            finite = np.isfinite(values)
            if not finite.any():
                continue
            values = values[finite]
            log_plot_t = log_plot_t[finite]
            label = name
            if "rpm" in name.lower():
                values = values / 60.0
                label = f"{name} [Hz]"
                has_rpm = True
            self.preview_ax2.plot(
                log_plot_t,
                values,
                lw=0.7,
                alpha=0.7,
                color=colors[idx],
                label=label,
            )
            labels.append(label)
        self.preview_ax2.set_ylabel("PWM / Hz" if has_rpm else "PWM")
        self.preview_ax2.grid(False)
        if labels:
            handles1, labels1 = ax_main.get_legend_handles_labels()
            handles2, labels2 = self.preview_ax2.get_legend_handles_labels()
            ax_main.legend(
                handles1 + handles2,
                labels1 + labels2,
                loc="upper right",
                fontsize=8,
                framealpha=0.5,
            )
    def _welch_spectrum(self, x, fs):
        x = signal.detrend(x, type="constant")
        if len(x) > analyze_vibro.MAX_WELCH_SAMPLES:
            step = int(np.ceil(len(x) / analyze_vibro.MAX_WELCH_SAMPLES))
            x = x[::step]
            fs = fs / step
        nperseg = max(1024, min(16384, int(fs * 2)))
        freqs, psd = signal.welch(
            x, fs=fs, window="hann", nperseg=nperseg, detrend="constant"
        )
        mag = np.sqrt(psd)
        if self._max_freq_hz:
            mask = freqs <= self._max_freq_hz
            freqs = freqs[mask]
            mag = mag[mask]
        return freqs, mag

    def _preview_axis_plots(self):
        path = self.file_var.get().strip()
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showerror("Missing file", f"File not found:\n{path}")
            return
        try:
            t, ax, ay, az = quantify_vibro.read_vibro_csv(path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        try:
            clip_g = self._parse_clip_g()
        except ValueError as exc:
            messagebox.showerror("Invalid G clip", str(exc))
            return

        try:
            seg = self._window_slice(
                t, self.start_var.get().strip(), self.end_var.get().strip()
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return

        t = t[seg]
        ax = ax[seg]
        ay = ay[seg]
        az = az[seg]
        ax, ay, az = self._apply_clip(ax, ay, az, clip_g)
        if len(t) < 4:
            messagebox.showerror("Error", "Not enough samples in selected window.")
            return
        seg_dt = np.median(np.diff(t))
        fs = 1.0 / seg_dt if seg_dt > 0 else 0.0
        if fs <= 0:
            messagebox.showerror("Error", "Invalid sample rate for selected window.")
            return

        win = tk.Toplevel(self)
        win.title("Axis spectra and spectrograms")
        fig = Figure(figsize=(10, 8), dpi=100)
        axes = fig.subplots(3, 2, sharex="col")
        readout = tk.Label(win, text="")
        readout.pack(anchor="w", padx=6, pady=4)
        ax_data = {}

        axis_data = [(ax, "aX [g]"), (ay, "aY [g]"), (az, "aZ [g]")]
        for row_idx, (data, label) in enumerate(axis_data):
            freqs, mag = self._welch_spectrum(data, fs)
            mag_db = 20 * np.log10(mag + 1e-12)
            ax_spec = axes[row_idx, 0]
            ax_spec.plot(freqs, mag_db, lw=1)
            ax_spec.set_ylabel(f"{label}\nMag [dB]")
            ax_spec.grid(True, alpha=0.3)
            if self._max_freq_hz:
                ax_spec.set_xlim(0, self._max_freq_hz)
            ax_data[ax_spec] = {
                "kind": "spectrum",
                "x": freqs,
                "y": mag_db,
                "label": label,
            }

            fs_spec = fs
            data_spec = data
            if len(data_spec) > analyze_vibro.MAX_SPECTRO_SAMPLES:
                step = int(np.ceil(len(data_spec) / analyze_vibro.MAX_SPECTRO_SAMPLES))
                data_spec = data_spec[::step]
                fs_spec = fs_spec / step
            nperseg = max(256, min(4096, int(fs_spec * 2)))
            noverlap = int(nperseg * 0.75)
            f, tt, sxx = signal.spectrogram(
                data_spec,
                fs=fs_spec,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                mode="magnitude",
            )
            sxx_db = 20 * np.log10(sxx + 1e-12)
            ax_sg = axes[row_idx, 1]
            pcm = ax_sg.pcolormesh(tt, f, sxx_db, shading="gouraud")
            ax_sg.set_ylabel(f"{label}\nFreq [Hz]")
            ax_sg.grid(False)
            if self._max_freq_hz:
                ax_sg.set_ylim(0, self._max_freq_hz)
            fig.colorbar(pcm, ax=ax_sg, pad=0.01, label="Mag [dB]")
            ax_data[ax_sg] = {
                "kind": "spectrogram",
                "t": tt,
                "f": f,
                "sxx_db": sxx_db,
                "label": label,
            }

        axes[-1, 0].set_xlabel("Frequency [Hz]")
        axes[-1, 1].set_xlabel("Time [s]")
        fig.suptitle(os.path.basename(path))
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        canvas.mpl_connect(
            "motion_notify_event",
            lambda event: self._on_axis_hover(event, ax_data, readout),
        )
        canvas.draw_idle()

    def _on_axis_hover(self, event, ax_data, readout):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            readout.config(text="")
            return
        info = ax_data.get(event.inaxes)
        if not info:
            readout.config(text="")
            return
        if info["kind"] == "spectrum":
            x = info["x"]
            y = info["y"]
            idx = int(np.clip(np.searchsorted(x, event.xdata), 0, len(x) - 1))
            readout.config(
                text=(
                    f"{info['label']} spectrum: f={x[idx]:.2f} Hz  "
                    f"mag={y[idx]:.2f} dB"
                )
            )
            return
        if info["kind"] == "spectrogram":
            tt = info["t"]
            ff = info["f"]
            sxx_db = info["sxx_db"]
            ti = int(np.clip(np.searchsorted(tt, event.xdata), 0, len(tt) - 1))
            fi = int(np.clip(np.searchsorted(ff, event.ydata), 0, len(ff) - 1))
            mag_db = sxx_db[fi, ti]
            readout.config(
                text=(
                    f"{info['label']} spectrogram: t={tt[ti]:.3f} s  "
                    f"f={ff[fi]:.2f} Hz  mag={mag_db:.2f} dB"
                )
            )
            return

    def _preview_pwm_plot(self):
        log_path = self.log_file_var.get().strip()
        if not log_path:
            messagebox.showerror("Missing log", "Please choose an ArduPilot log file.")
            return
        if not os.path.exists(log_path):
            messagebox.showerror("Missing log", f"Log file not found:\n{log_path}")
            return
        try:
            log_t, series = self._read_ardupilot_log(log_path)
        except Exception as exc:
            messagebox.showerror("Log error", str(exc))
            return

        offset = self._parse_log_offset()
        log_t = log_t + offset

        win = tk.Toplevel(self)
        win.title("PWM / RPM vs time")
        fig = Figure(figsize=(9, 4.5), dpi=100)
        ax = fig.add_subplot(111)
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
        ]
        labels = []
        has_rpm = False
        prefer = ["RCOU.C1", "RCOU.C2", "RCOU.C3", "RCOU.C4"]
        if any(name in series for name in prefer):
            ordered = [(name, series[name]) for name in prefer if name in series]
        else:
            ordered = list(series.items())

        active_baseline = 1100.0
        active_tol = 5.0
        if ordered:
            stacked = np.vstack([vals for _, vals in ordered])
            finite_any = np.isfinite(stacked).any(axis=0)
            active = np.any(np.abs(stacked - active_baseline) > active_tol, axis=0)
            active = active & finite_any
            if active.any():
                start_idx = int(np.argmax(active))
                log_t = log_t[start_idx:]
                for name in list(series.keys()):
                    series[name] = series[name][start_idx:]
                ordered = [(name, series[name]) for name, _ in ordered]

        t_min = None
        t_max = None
        path = self.file_var.get().strip()
        if path and os.path.exists(path):
            try:
                t, _, _, _ = quantify_vibro.read_vibro_csv(path)
                t_min = float(t[0]) if len(t) else None
                t_max = float(t[-1]) if len(t) else None
            except Exception:
                t_min = None
                t_max = None
        if t_min is not None and t_max is not None:
            mask = (log_t >= t_min) & (log_t <= t_max)
        else:
            mask = None
        for idx, (name, values) in enumerate(ordered):
            if idx >= len(colors):
                break
            if mask is not None and mask.any():
                plot_t = log_t[mask]
                plot_v = values[mask]
            else:
                plot_t = log_t
                plot_v = values
            finite = np.isfinite(plot_v)
            if not finite.any():
                continue
            plot_t = plot_t[finite]
            plot_v = plot_v[finite]
            label = name
            if "rpm" in name.lower():
                plot_v = plot_v / 60.0
                label = f"{name} [Hz]"
                has_rpm = True
            ax.plot(plot_t, plot_v, lw=0.8, color=colors[idx], label=label)
            labels.append(label)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PWM / Hz" if has_rpm else "PWM")
        ax.grid(True, alpha=0.3)
        if labels:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.5)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        canvas.draw_idle()


def main():
    app = VibroGui()
    app.mainloop()


if __name__ == "__main__":
    main()
