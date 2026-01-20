#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import analyze_vibro
import quantify_vibro
import numpy as np
from scipy import signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a CSV to begin.")

        self._build_ui()
        self._preview_cid = None
        self._preview_data = None

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        file_frame = tk.Frame(self)
        file_frame.pack(fill="x", **pad)
        tk.Label(file_frame, text="CSV file:").pack(side="left")
        tk.Entry(file_frame, textvariable=self.file_var, width=60).pack(
            side="left", padx=6, fill="x", expand=True
        )
        tk.Button(file_frame, text="Browse", command=self._pick_file).pack(side="left")

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
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=plot_frame)
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
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
            )
            metrics = quantify_vibro.quantify_file(path, start, end)
            metrics_text = _format_metrics(metrics)
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

        mag = (ax * ax + ay * ay + az * az) ** 0.5
        self.preview_ax.clear()
        self.preview_ax.plot(t, mag, lw=0.8)
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


def main():
    app = VibroGui()
    app.mainloop()


if __name__ == "__main__":
    main()
