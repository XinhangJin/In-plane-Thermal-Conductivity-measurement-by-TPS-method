#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graphical front end for TPS_Fitting_CPU_Kernel_ABC.py.

Put this file in the same folder as TPS_Fitting_CPU_Kernel_ABC.py, then run:
    python main.py

Dependencies:
    numpy pandas scipy openpyxl matplotlib
Tkinter is included with most standard Python installations.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from TPS_Fitting_CPU_Kernel_ABC import fit_tps_kernel, export_kernel_result

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


APP_TITLE = "TPS In-Plane Thermal Conductivity Fitting"
DEFAULT_A = 160372.0
DEFAULT_B = 218.0
DEFAULT_C = -0.03


class TPSFittingGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1220x820")
        self.minsize(1040, 700)

        self.input_path: Path | None = None
        self.input_dataframe: pd.DataFrame | None = None
        self.result: dict | None = None

        self._build_variables()
        self._build_ui()

    def _build_variables(self) -> None:
        self.file_var = tk.StringVar()
        self.sheet_var = tk.StringVar(value=" ")
        self.time_col_var = tk.StringVar(value="0")
        self.temp_col_var = tk.StringVar(value="1")

        self.p0_var = tk.StringVar(value="20")
        self.cv_var = tk.StringVar(value="2.0")
        self.r_var = tk.StringVar(value="2.0")
        self.h_var = tk.StringVar(value="100")
        self.t_end_var = tk.StringVar(value="")

        self.a_var = tk.StringVar(value=str(DEFAULT_A))
        self.b_var = tk.StringVar(value=str(DEFAULT_B))
        self.c_var = tk.StringVar(value=str(DEFAULT_C))

        self.ref_var = tk.StringVar(value="20")
        self.lambda_min_var = tk.StringVar(value="0.01")
        self.lambda_max_var = tk.StringVar(value="50")
        self.max_points_var = tk.StringVar(value="200")
        self.sigma_points_var = tk.StringVar(value="1200")
        self.dtype_var = tk.StringVar(value="float64")

        self.status_var = tk.StringVar(value="Ready")
        self.lambda_hat_var = tk.StringVar(value="—")
        self.lambda_final_var = tk.StringVar(value="—")
        self.error_var = tk.StringVar(value="—")
        self.srel_var = tk.StringVar(value="—")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_input_group(left)
        self._build_experiment_group(left)
        self._build_correction_group(left)
        self._build_advanced_group(left)
        self._build_actions(left)
        self._build_results(right)
        self._build_plot(right)

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken")
        status.pack(fill="x", side="bottom")

    @staticmethod
    def _entry_row(parent, row: int, label: str, variable: tk.StringVar, unit: str = "") -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=18)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        if unit:
            ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", padx=(0, 5), pady=3)
        return entry

    def _build_input_group(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="1. Time–Temperature Data", padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Data file").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(box, textvariable=self.file_var, width=37, state="readonly").grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=3
        )
        ttk.Button(box, text="Browse…", command=self.select_file).grid(row=0, column=3, padx=5, pady=3)

        ttk.Label(box, text="Excel sheet").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(box, textvariable=self.sheet_var, width=18).grid(row=1, column=1, sticky="w", padx=5, pady=3)
        ttk.Label(box, text="CSV ignores sheet").grid(row=1, column=2, columnspan=2, sticky="w", padx=5)

        ttk.Label(box, text="Time column").grid(row=2, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(box, textvariable=self.time_col_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(box, text="index/name").grid(row=2, column=2, sticky="w")

        ttk.Label(box, text="Temperature column").grid(row=3, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(box, textvariable=self.temp_col_var, width=10).grid(row=3, column=1, sticky="w", padx=5)
        ttk.Label(box, text="index/name").grid(row=3, column=2, sticky="w")

        ttk.Button(box, text="Preview data", command=self.preview_data).grid(
            row=4, column=0, columnspan=4, sticky="ew", padx=5, pady=(8, 3)
        )

    def _build_experiment_group(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="2. Experiment Parameters", padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        self._entry_row(box, 0, "Heating power P0", self.p0_var, "mW")
        self._entry_row(box, 1, "Volumetric heat capacity Cv", self.cv_var, "J cm⁻³ K⁻¹")
        self._entry_row(box, 2, "Probe radius r", self.r_var, "mm")
        self._entry_row(box, 3, "Film thickness h", self.h_var, "µm")
        self._entry_row(box, 4, "Fit end time", self.t_end_var, "s (blank = max)")

    def _build_correction_group(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="3. Thickness Correction", padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        self._entry_row(box, 0, "A", self.a_var)
        self._entry_row(box, 1, "B", self.b_var, "µm")
        self._entry_row(box, 2, "C", self.c_var, "W m⁻¹ K⁻¹")
        ttk.Label(box, text="Error(h) = A / (h + B)² + C").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 0)
        )

    def _build_advanced_group(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="4. Advanced", padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        self._entry_row(box, 0, "Fitting Start Data Point", self.ref_var)
        self._entry_row(box, 1, "λ minimum", self.lambda_min_var, "W m⁻¹ K⁻¹")
        self._entry_row(box, 2, "λ maximum", self.lambda_max_var, "W m⁻¹ K⁻¹")
        self._entry_row(box, 3, "Max points", self.max_points_var)
        self._entry_row(box, 4, "Sigma points", self.sigma_points_var)

        ttk.Label(box, text="CPU dtype").grid(row=5, column=0, sticky="w", padx=5, pady=3)
        ttk.Combobox(box, textvariable=self.dtype_var, values=("float64", "float32"), width=15,
                     state="readonly").grid(row=5, column=1, sticky="w", padx=5, pady=3)

    def _build_actions(self, parent) -> None:
        box = ttk.Frame(parent)
        box.pack(fill="x")
        box.columnconfigure((0, 1), weight=1)

        self.fit_button = ttk.Button(box, text="Start fitting", command=self.start_fitting)
        self.fit_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        ttk.Button(box, text="Export parameters", command=self.export_parameters).grid(
            row=1, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(box, text="Export T–T data", command=self.export_time_temperature).grid(
            row=1, column=1, sticky="ew", padx=2, pady=2
        )

    def _build_results(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Fitting Results", padding=10)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for i in range(4):
            box.columnconfigure(i, weight=1)

        items = [
            ("λ_Fit (Before Thickness Correction)", self.lambda_hat_var, "W m⁻¹ K⁻¹"),
            ("λ_Finally", self.lambda_final_var, "W m⁻¹ K⁻¹"),
            ("Relative fit error", self.srel_var, "%"),
        ]
        for col, (label, variable, unit) in enumerate(items):
            ttk.Label(box, text=label, anchor="center").grid(row=0, column=col, sticky="ew", padx=6)
            ttk.Label(box, textvariable=variable, font=("TkDefaultFont", 15, "bold"), anchor="center").grid(
                row=1, column=col, sticky="ew", padx=6, pady=4
            )
            ttk.Label(box, text=unit, anchor="center").grid(row=2, column=col, sticky="ew", padx=6)

    def _build_plot(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Time–Temperature / Fitted Curve", padding=6)
        box.grid(row=1, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(7.5, 5.5), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("ΔT (K)")
            self.ax.grid(True, alpha=0.25)
            self.canvas = FigureCanvasTkAgg(self.figure, master=box)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            toolbar = NavigationToolbar2Tk(self.canvas, box, pack_toolbar=False)
            toolbar.update()
            toolbar.grid(row=1, column=0, sticky="ew")
        else:
            ttk.Label(
                box,
                text="Matplotlib is not available. Fitting still works, but the curve cannot be plotted.\n"
                     "Install it with: pip install matplotlib",
                justify="center",
            ).grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def select_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select time–temperature data",
            filetypes=[
                ("Excel / CSV", "*.xlsx *.xls *.csv *.txt"),
                ("Excel", "*.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return
        self.input_path = Path(filename)
        self.file_var.set(str(self.input_path))
        self.status_var.set(f"Selected: {self.input_path.name}")

    def _read_input_file(self) -> pd.DataFrame:
        if self.input_path is None:
            text = self.file_var.get().strip()
            if not text:
                raise ValueError("Please select an input Excel/CSV file.")
            self.input_path = Path(text)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        suffix = self.input_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            sheet = self.sheet_var.get().strip()
            sheet_arg = sheet if sheet else 0
            return pd.read_excel(self.input_path, sheet_name=sheet_arg)
        if suffix in {".csv", ".txt"}:
            try:
                return pd.read_csv(self.input_path)
            except Exception:
                return pd.read_csv(self.input_path, sep=None, engine="python")
        raise ValueError("Only .xlsx, .xls, .csv and .txt input files are supported.")

    @staticmethod
    def _resolve_column(frame: pd.DataFrame, spec: str):
        spec = spec.strip()
        if spec == "":
            raise ValueError("Time/temperature column cannot be blank.")
        if spec in frame.columns:
            return spec
        try:
            index = int(spec)
        except ValueError:
            index = None
        if index is not None:
            if index < 0 or index >= frame.shape[1]:
                raise IndexError(f"Column index {index} is outside 0..{frame.shape[1] - 1}.")
            return frame.columns[index]
        matches = [col for col in frame.columns if str(col).strip().casefold() == spec.casefold()]
        if matches:
            return matches[0]
        raise KeyError(f"Column '{spec}' was not found. Available columns: {list(frame.columns)}")

    def _load_time_temperature(self) -> tuple[np.ndarray, np.ndarray]:
        frame = self._read_input_file()
        time_col = self._resolve_column(frame, self.time_col_var.get())
        temp_col = self._resolve_column(frame, self.temp_col_var.get())

        data = pd.DataFrame({
            "time_s": pd.to_numeric(frame[time_col], errors="coerce"),
            "temperature_K": pd.to_numeric(frame[temp_col], errors="coerce"),
        }).dropna()
        data = data[np.isfinite(data["time_s"]) & np.isfinite(data["temperature_K"])]
        if data.empty:
            raise ValueError("No valid numeric time–temperature rows were found.")
        self.input_dataframe = data.reset_index(drop=True)
        return (
            self.input_dataframe["time_s"].to_numpy(dtype=float),
            self.input_dataframe["temperature_K"].to_numpy(dtype=float),
        )

    def preview_data(self) -> None:
        try:
            time_s, temperature = self._load_time_temperature()
        except Exception as exc:
            messagebox.showerror("Data error", str(exc))
            return

        window = tk.Toplevel(self)
        window.title("Input data preview")
        window.geometry("640x450")

        tree = ttk.Treeview(window, columns=("i", "time", "temp"), show="headings")
        tree.heading("i", text="#")
        tree.heading("time", text="Time (s)")
        tree.heading("temp", text="Temperature")
        tree.column("i", width=60, anchor="center")
        tree.column("time", width=220, anchor="e")
        tree.column("temp", width=220, anchor="e")
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        for i, (t, temp) in enumerate(zip(time_s[:300], temperature[:300]), start=1):
            tree.insert("", "end", values=(i, f"{t:.8g}", f"{temp:.8g}"))

        ttk.Label(window, text=f"Valid rows: {len(time_s)}; previewing first {min(len(time_s), 300)} rows").pack(
            anchor="w", padx=8, pady=(0, 8)
        )

    @staticmethod
    def _float(text: str, name: str, allow_blank: bool = False):
        text = text.strip()
        if allow_blank and text == "":
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    @staticmethod
    def _int(text: str, name: str) -> int:
        try:
            value = int(text.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        return value

    def _collect_parameters(self) -> dict:
        return {
            "P0": self._float(self.p0_var.get(), "P0"),
            "Cv": self._float(self.cv_var.get(), "Cv"),
            "r": self._float(self.r_var.get(), "r"),
            "h": self._float(self.h_var.get(), "h"),
            "t_end": self._float(self.t_end_var.get(), "Fit end time", allow_blank=True),
            "A": self._float(self.a_var.get(), "A"),
            "B": self._float(self.b_var.get(), "B"),
            "C": self._float(self.c_var.get(), "C"),
            "reference_index": self._int(self.ref_var.get(), "Reference index"),
            "lambda_min": self._float(self.lambda_min_var.get(), "λ minimum"),
            "lambda_max": self._float(self.lambda_max_var.get(), "λ maximum"),
            "max_points": self._int(self.max_points_var.get(), "Max points"),
            "sigma_points": self._int(self.sigma_points_var.get(), "Sigma points"),
            "cpu_dtype": self.dtype_var.get(),
            "return_time_temperature": True,
        }

    def start_fitting(self) -> None:
        try:
            time_s, temperature = self._load_time_temperature()
            params = self._collect_parameters()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))
            return

        self.fit_button.configure(state="disabled")
        self.status_var.set("Fitting…")
        self.update_idletasks()

        thread = threading.Thread(
            target=self._fit_worker,
            args=(time_s, temperature, params),
            daemon=True,
        )
        thread.start()

    def _fit_worker(self, time_s: np.ndarray, temperature: np.ndarray, params: dict) -> None:
        try:
            result = fit_tps_kernel(time_s, temperature, **params)
        except Exception as exc:
            details = traceback.format_exc()
            self.after(0, self._fit_failed, exc, details)
            return
        self.after(0, self._fit_finished, result)

    def _fit_failed(self, exc: Exception, details: str) -> None:
        self.fit_button.configure(state="normal")
        self.status_var.set("Fitting failed")
        messagebox.showerror("Fitting failed", f"{exc}\n\nDetails:\n{details[-2500:]}")

    def _fit_finished(self, result: dict) -> None:
        self.result = result
        self.fit_button.configure(state="normal")
        self.lambda_hat_var.set(f"{result['lambda_hat_W_mK']:.6g}")
        self.lambda_final_var.set(f"{result['lambda_corrected_W_mK']:.6g}")
        self.error_var.set(f"{result['thickness_error_W_mK']:.6g}")
        self.srel_var.set(f"{result['relative_fit_error']:.6g}")
        self.status_var.set("Fitting completed")
        self._update_plot()

    def _update_plot(self) -> None:
        if not MATPLOTLIB_AVAILABLE or self.result is None:
            return
        data = self.result.get("time_temperature")
        if data is None or len(data) == 0:
            return

        self.ax.clear()
        self.ax.plot(data["time_s"], data["experimental_delta_T_K"], marker="o", markersize=3,
                     linewidth=1.2, label="Experimental ΔT")
        self.ax.plot(data["time_s"], data["calculated_delta_T_K"], linewidth=2.0,
                     label="TPS fitted ΔT")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("ΔT (K)")
        self.ax.set_title(
            f"λ_hat = {self.result['lambda_hat_W_mK']:.4g} W m⁻¹ K⁻¹   |   "
            f"λ_corrected = {self.result['lambda_corrected_W_mK']:.4g} W m⁻¹ K⁻¹"
        )
        self.ax.grid(True, alpha=0.25)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _require_result(self) -> bool:
        if self.result is None:
            messagebox.showwarning("No result", "Run a fitting calculation first.")
            return False
        return True

    def export_parameters(self) -> None:
        if not self._require_result():
            return
        filename = filedialog.asksaveasfilename(
            title="Export fitting parameters/results",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="TPS_fitting_result.csv",
        )
        if not filename:
            return
        try:
            export_kernel_result(self.result, parameter_path=filename)
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return
        self.status_var.set(f"Parameters exported: {Path(filename).name}")

    def export_time_temperature(self) -> None:
        if not self._require_result():
            return
        filename = filedialog.asksaveasfilename(
            title="Export time–temperature data",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="TPS_time_temperature_fit.csv",
        )
        if not filename:
            return
        try:
            export_kernel_result(self.result, time_temperature_path=filename)
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return
        self.status_var.set(f"Time–temperature data exported: {Path(filename).name}")


def main() -> None:
    app = TPSFittingGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
