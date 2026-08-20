#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPS CPU batch fitting — same-folder version with thickness correction.

Place this script, the parameter workbook, and all experiment workbooks
in the same folder, then run:

    python TPS_Fitting_CPU_With_Correction.py

Default parameter workbook:
    Calculation Parameters.xlsx

Use another parameter workbook:
    python TPS_Fitting_CPU_With_Correction.py ^
        --parameter-table "Calculation Parameters Al2O3.xlsx"

Replace an existing result file:
    python TPS_Fitting_CPU_With_Correction.py --overwrite

Save fitted curves:
    python TPS_Fitting_CPU_With_Correction.py --save-curves

Outputs:
    All_fitting_data_CPU_corrected.csv
    Fit_curves_CPU/

Thickness-correction model:
    error_h = A / (h_um + B_um)^2 + C
    lambda_corrected = lambda_hat - error_h

Default correction parameters:
    A     = 160372
    B_um  = 218
    C     = -0.03

Therefore:
    lambda_corrected
      = lambda_hat - 160372/(h_um + 218)^2 + 0.03
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import i0e


LOGGER = logging.getLogger("tps_cpu")
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# Thickness-correction settings
# ============================================================

CORRECTION_A_DEFAULT = 160372.0
CORRECTION_B_UM_DEFAULT = 218.0
CORRECTION_C_DEFAULT = -0.03
MIN_CORRECTED_LAMBDA_DEFAULT = 0.01


# ============================================================
# Model options
# ============================================================

@dataclass
class Options:
    P0: float
    Cv: float
    r: float
    h: float
    t_end: float

    m: int = 20
    sigma_min: float = 1.0e-3
    image_terms: int = 150
    reference_index: int = 20          # one-based

    lambda_min: float = 0.01
    lambda_max: float = 50.0

    max_points: int = 200
    sigma_points: int = 1200
    cpu_dtype: str = "float64"

    def validate(self) -> None:
        values = {
            "P0": self.P0,
            "Cv": self.Cv,
            "r": self.r,
            "h": self.h,
            "t_end": self.t_end,
            "sigma_min": self.sigma_min,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
        }

        for name, value in values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be positive and finite; got {value!r}"
                )

        if self.lambda_min >= self.lambda_max:
            raise ValueError(
                "lambda_min must be smaller than lambda_max"
            )

        if self.m < 1:
            raise ValueError("m must be a positive integer")

        if self.image_terms < 1:
            raise ValueError(
                "image_terms must be a positive integer"
            )

        if self.reference_index < 1:
            raise ValueError(
                "reference_index must be >= 1"
            )

        if self.max_points < self.reference_index + 2:
            raise ValueError(
                "max_points is too small for the selected reference_index"
            )

        if self.sigma_points < 100:
            raise ValueError(
                "sigma_points must be >= 100"
            )

        if self.cpu_dtype not in {"float32", "float64"}:
            raise ValueError(
                "cpu_dtype must be float32 or float64"
            )


def check_cpu() -> None:
    LOGGER.info(
        "CPU mode ready | NumPy %s",
        np.__version__,
    )


# ============================================================
# TPS kernel
# ============================================================

class TPSCPUKernel:
    """
    CPU lookup table for D(tau), constructed once per experiment.
    """

    def __init__(
        self,
        opts: Options,
        max_time_s: float,
    ):
        opts.validate()

        self.opts = opts
        self.dtype = (
            np.float32
            if opts.cpu_dtype == "float32"
            else np.float64
        )

        tau_max = (
            np.sqrt(
                opts.lambda_max
                * max_time_s
                / opts.Cv
            )
            / opts.r
        )

        sigma_lo = max(
            opts.sigma_min,
            1.0e-6,
        )

        sigma_hi = max(
            float(tau_max),
            sigma_lo * 50.0,
        )

        self.sigma_lo = sigma_lo

        self.sigma = np.logspace(
            np.log10(self.dtype(sigma_lo)),
            np.log10(self.dtype(sigma_hi)),
            opts.sigma_points,
            dtype=self.dtype,
        )

        sigma2 = self.sigma * self.sigma

        ring = np.arange(
            1,
            opts.m + 1,
            dtype=self.dtype,
        )

        l_idx = ring[:, None]
        k_idx = ring[None, :]

        lk = l_idx * k_idx
        lmk2 = (l_idx - k_idx) ** 2

        inv_4m2 = self.dtype(
            1.0 / (4.0 * opts.m * opts.m)
        )

        inv_2m2 = self.dtype(
            1.0 / (2.0 * opts.m * opts.m)
        )

        exponent_floor = (
            -80.0
            if self.dtype == np.float32
            else -745.0
        )

        sigma3 = sigma2[:, None, None]

        exp_core = np.exp(
            np.clip(
                -lmk2[None, :, :]
                * inv_4m2
                / sigma3,
                exponent_floor,
                0.0,
            )
        )

        bessel_arg = (
            lk[None, :, :]
            * inv_2m2
            / sigma3
        )

        s1 = np.sum(
            lk[None, :, :]
            * exp_core
            * i0e(bessel_arg),
            axis=(1, 2),
        )

        image_idx = np.arange(
            1,
            opts.image_terms + 1,
            dtype=self.dtype,
        )

        hr2 = self.dtype(
            (opts.h / opts.r) ** 2
        )

        exponent = (
            -(image_idx[None, :] ** 2)
            * hr2
            / sigma2[:, None]
        )

        s2 = np.sum(
            np.exp(
                np.clip(
                    exponent,
                    exponent_floor,
                    0.0,
                )
            ),
            axis=1,
        )

        tiny = np.finfo(self.dtype).tiny

        integrand = (
            s1
            * (1.0 + 2.0 * s2)
            / np.maximum(sigma2, tiny)
        )

        integrand = np.where(
            np.isfinite(integrand),
            integrand,
            self.dtype(0.0),
        )

        ds = (
            self.sigma[1:]
            - self.sigma[:-1]
        )

        trapezoids = (
            self.dtype(0.5)
            * (
                integrand[1:]
                + integrand[:-1]
            )
            * ds
        )

        self.cumulative = np.concatenate(
            [
                np.zeros(
                    1,
                    dtype=self.dtype,
                ),
                np.cumsum(trapezoids),
            ]
        )

        self.normalization = self.dtype(
            (
                1.0
                / (
                    opts.m
                    * (opts.m + 1)
                )
            ) ** 2
        )

    def d_of_tau_batch(
        self,
        tau: np.ndarray,
    ) -> np.ndarray:
        flat = tau.ravel()

        interpolated = np.interp(
            flat,
            self.sigma,
            self.cumulative,
        ).reshape(tau.shape)

        return np.where(
            tau >= self.sigma_lo,
            interpolated
            * self.normalization,
            self.dtype(0.0),
        )


# ============================================================
# Experimental-data preparation
# ============================================================

def prepare_experiment(
    experiment_data: pd.DataFrame | np.ndarray,
    opts: Options,
) -> tuple[np.ndarray, np.ndarray, int]:

    if isinstance(
        experiment_data,
        (pd.DataFrame, pd.Series),
    ):
        experiment_data = (
            experiment_data.to_numpy()
        )

    values = np.asarray(experiment_data)

    if (
        values.ndim != 2
        or values.shape[1] < 2
    ):
        raise ValueError(
            "Experiment data must contain "
            "time and temperature columns"
        )

    frame = pd.DataFrame(
        {
            "time": pd.to_numeric(
                values[:, 0],
                errors="coerce",
            ),
            "temperature": pd.to_numeric(
                values[:, 1],
                errors="coerce",
            ),
        }
    ).dropna()

    frame = frame[
        np.isfinite(frame["time"])
        & np.isfinite(frame["temperature"])
        & (frame["time"] >= 0)
        & (frame["time"] <= opts.t_end)
    ]

    frame = (
        frame
        .sort_values("time")
        .drop_duplicates("time")
        .iloc[: opts.max_points]
        .reset_index(drop=True)
    )

    minimum_points = (
        opts.reference_index + 2
    )

    if len(frame) < minimum_points:
        raise ValueError(
            f"At least {minimum_points} valid points "
            f"are required; got {len(frame)}"
        )

    time_s = frame[
        "time"
    ].to_numpy(
        dtype=np.float64
    )

    temperature = frame[
        "temperature"
    ].to_numpy(
        dtype=np.float64
    )

    reference = (
        opts.reference_index - 1
    )

    delta_temperature = (
        temperature
        - temperature[reference]
    )

    return (
        time_s,
        delta_temperature,
        reference,
    )


# ============================================================
# Lambda-grid tools
# ============================================================

def make_grid(
    start: float,
    stop: float,
    step: float,
) -> np.ndarray:

    if stop < start:
        return np.empty(
            0,
            dtype=np.float64,
        )

    count = (
        int(
            np.floor(
                (stop - start) / step
            )
        )
        + 1
    )

    return np.round(
        start
        + np.arange(
            count,
            dtype=np.float64,
        )
        * step,
        10,
    )


def make_clamped_grid(
    center: float,
    half_span: float,
    step: float,
    lower: float,
    upper: float,
) -> np.ndarray:

    return make_grid(
        max(
            center - half_span,
            lower,
        ),
        min(
            center + half_span,
            upper,
        ),
        step,
    )


def evaluate_lambda_grid_cpu(
    lambdas: np.ndarray,
    time_s: np.ndarray,
    experimental_delta_t: np.ndarray,
    reference: int,
    opts: Options,
    kernel: TPSCPUKernel,
    return_best_curve: bool = False,
) -> tuple[
    np.ndarray,
    Optional[np.ndarray],
    int,
]:

    if lambdas.size == 0:
        raise ValueError(
            "Lambda grid is empty"
        )

    dtype = kernel.dtype

    lambda_cpu = np.asarray(
        lambdas,
        dtype=dtype,
    )[:, None]

    time_cpu = np.asarray(
        time_s,
        dtype=dtype,
    )[None, :]

    experiment_cpu = np.asarray(
        experimental_delta_t,
        dtype=dtype,
    )[None, :]

    alpha = (
        lambda_cpu
        / dtype(opts.Cv)
    )

    tau = (
        np.sqrt(
            np.maximum(
                alpha * time_cpu,
                dtype(0.0),
            )
        )
        / dtype(opts.r)
    )

    d_tau = kernel.d_of_tau_batch(
        tau
    )

    coefficient = (
        dtype(opts.P0)
        / (
            dtype(np.pi ** 1.5)
            * dtype(opts.r)
            * lambda_cpu
        )
    )

    calculated = (
        coefficient * d_tau
    )

    calculated = (
        calculated
        - calculated[
            :,
            reference : reference + 1
        ]
    )

    fit_start = min(
        20,
        calculated.shape[1] - 1,
    )

    calculated_sub = calculated[
        :,
        fit_start:,
    ]

    experiment_sub = experiment_cpu[
        :,
        fit_start:,
    ]

    difference = (
        experiment_sub
        - calculated_sub
    )

    denominator = np.sum(
        experiment_sub
        * experiment_sub
    )

    tiny = np.finfo(dtype).tiny

    errors = np.where(
        denominator > tiny,
        np.sqrt(
            np.sum(
                difference
                * difference,
                axis=1,
            )
            / denominator
        ),
        np.sqrt(
            np.mean(
                difference
                * difference,
                axis=1,
            )
        ),
    )

    best_index = int(
        np.argmin(errors)
    )

    errors_cpu = np.asarray(
        errors
    )

    best_curve = None

    if return_best_curve:
        best_curve = np.asarray(
            calculated[best_index]
        )

    return (
        errors_cpu,
        best_curve,
        best_index,
    )


def predict_lambda_grid_cpu(
    experiment_data: pd.DataFrame | np.ndarray,
    opts: Options,
) -> tuple[float, dict]:

    opts.validate()

    (
        time_s,
        te,
        reference,
    ) = prepare_experiment(
        experiment_data,
        opts,
    )

    kernel = TPSCPUKernel(
        opts,
        float(
            np.max(time_s)
        ),
    )

    # Coarse search: 1 W m-1 K-1
    grid1 = make_grid(
        opts.lambda_min,
        opts.lambda_max,
        1.0,
    )

    (
        errors1,
        _,
        index1,
    ) = evaluate_lambda_grid_cpu(
        grid1,
        time_s,
        te,
        reference,
        opts,
        kernel,
    )

    best1 = float(
        grid1[index1]
    )

    # Medium search: 0.1 W m-1 K-1
    grid2 = make_clamped_grid(
        best1,
        1.0,
        0.1,
        opts.lambda_min,
        opts.lambda_max,
    )

    (
        errors2,
        _,
        index2,
    ) = evaluate_lambda_grid_cpu(
        grid2,
        time_s,
        te,
        reference,
        opts,
        kernel,
    )

    best2 = float(
        grid2[index2]
    )

    # Fine search: 0.01 W m-1 K-1
    grid3 = make_clamped_grid(
        best2,
        0.1,
        0.01,
        opts.lambda_min,
        opts.lambda_max,
    )

    (
        errors3,
        best_curve,
        index3,
    ) = evaluate_lambda_grid_cpu(
        grid3,
        time_s,
        te,
        reference,
        opts,
        kernel,
        return_best_curve=True,
    )

    lambda_hat = float(
        grid3[index3]
    )

    report = {
        "lambda_hat": lambda_hat,
        "s3_min": float(
            100*errors3[index3]
        ),
        "t": time_s,
        "Te": te,
        "Tc": best_curve,
    }

    return (
        lambda_hat,
        report,
    )


# ============================================================
# Unit normalization
# ============================================================

def norm_cv(
    value: object,
) -> float:
    """
    Convert Cv to J m-3 K-1.

    Input:
      - values <= 10 are interpreted as J cm-3 K-1
      - larger values are interpreted as J m-3 K-1
    """
    number = float(value)

    return (
        number * 1.0e6
        if number <= 10.0
        else number
    )


def norm_h(
    value: object,
) -> float:
    """
    Convert thickness to metres.

    Input:
      - values > 1 are interpreted as micrometres
      - values <= 1 are interpreted as metres
    """
    number = float(value)

    return (
        number * 1.0e-6
        if number > 1.0
        else number
    )


def norm_r(
    value: object,
) -> float:
    """
    Convert probe radius to metres.

    Input:
      - values > 0.05 are interpreted as millimetres
      - values <= 0.05 are interpreted as metres
    """
    number = float(value)

    return (
        number * 1.0e-3
        if number > 0.05
        else number
    )


def norm_p0(
    value: object,
) -> float:
    """
    Convert heating power to watts.

    Input:
      - values >= 1 are interpreted as milliwatts
      - values < 1 are interpreted as watts
    """
    number = float(value)

    return (
        number / 1000.0
        if number >= 1.0
        else number
    )


# ============================================================
# Public kernel API
# ============================================================

def fit_tps_kernel(
    time_s,
    temperature_k,
    *,
    P0,
    Cv,
    r,
    h,
    t_end=None,
    A=CORRECTION_A_DEFAULT,
    B=CORRECTION_B_UM_DEFAULT,
    C=CORRECTION_C_DEFAULT,
    reference_index=20,
    lambda_min=0.01,
    lambda_max=50.0,
    max_points=200,
    sigma_points=1200,
    cpu_dtype="float64",
    min_corrected_lambda=MIN_CORRECTED_LAMBDA_DEFAULT,
    return_time_temperature=True,
):
    """Public TPS fitting kernel.

    Parameters P0, Cv, r and h accept the same practical units as the batch
    interface: P0 in mW (or W if <1), Cv in J cm-3 K-1 (or J m-3 K-1
    if >10), r in mm (or m if <=0.05), and h in um (or m if <=1).

    Thickness correction:
        error_h = A / (h_um + B)^2 + C
        lambda_corrected = lambda_hat - error_h

    A, B and C are user-overridable. B is expressed in micrometres.
    The returned dictionary is intended as the stable integration interface
    for GUI, DLL/service wrappers, notebooks, or other Python programs.
    """
    time_arr = np.asarray(time_s, dtype=np.float64).reshape(-1)
    temp_arr = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
    if time_arr.size != temp_arr.size:
        raise ValueError("time_s and temperature_k must have the same length")
    if time_arr.size == 0:
        raise ValueError("time_s and temperature_k cannot be empty")

    if t_end is None:
        finite_time = time_arr[np.isfinite(time_arr)]
        if finite_time.size == 0:
            raise ValueError("time_s contains no finite values")
        t_end = float(np.max(finite_time))

    experiment = pd.DataFrame({
        "Time": time_arr,
        "Temperature": temp_arr,
    })

    opts = Options(
        P0=norm_p0(P0),
        Cv=norm_cv(Cv),
        r=norm_r(r),
        h=norm_h(h),
        t_end=float(t_end),
        reference_index=int(reference_index),
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        max_points=int(max_points),
        sigma_points=int(sigma_points),
        cpu_dtype=str(cpu_dtype),
    )

    lambda_hat, report = predict_lambda_grid_cpu(experiment, opts)
    h_um, thickness_error, lambda_corrected = calculate_thickness_correction(
        lambda_hat=lambda_hat,
        h_m=opts.h,
        correction_a=float(A),
        correction_b_um=float(B),
        correction_c=float(C),
        min_corrected_lambda=float(min_corrected_lambda),
    )

    result = {
        "lambda_hat_W_mK": float(lambda_hat),
        "lambda_corrected_W_mK": float(lambda_corrected),
        "thickness_error_W_mK": float(thickness_error),
        "relative_fit_error": float(report["s3_min"]),
        "A": float(A),
        "B_um": float(B),
        "C_W_mK": float(C),
        "h_um": float(h_um),
        "P0_mW": float(1.0e3 * opts.P0),
        "Cv_J_cm3_K": float(1.0e-6 * opts.Cv),
        "r_mm": float(1.0e3 * opts.r),
        "t_end_s": float(opts.t_end),
    }

    if return_time_temperature:
        prepared = pd.DataFrame({
            "time_s": time_arr,
            "temperature_K": temp_arr,
        })
        prepared = prepared[
            np.isfinite(prepared["time_s"])
            & np.isfinite(prepared["temperature_K"])
            & (prepared["time_s"] >= 0)
            & (prepared["time_s"] <= opts.t_end)
        ]
        prepared = (
            prepared.sort_values("time_s")
            .drop_duplicates("time_s")
            .iloc[: opts.max_points]
            .reset_index(drop=True)
        )
        reference_zero = opts.reference_index - 1
        reference_temperature = float(prepared["temperature_K"].iloc[reference_zero])

        result["reference_temperature_K"] = reference_temperature
        result["time_temperature"] = pd.DataFrame({
            "time_s": np.asarray(report["t"], dtype=np.float64),
            "temperature_K": prepared["temperature_K"].to_numpy(dtype=np.float64),
            "experimental_delta_T_K": np.asarray(report["Te"], dtype=np.float64),
            "calculated_delta_T_K": np.asarray(report["Tc"], dtype=np.float64),
        })

    return result


def export_kernel_result(result, parameter_path=None, time_temperature_path=None):
    """Optional output adapter for the public kernel API.

    parameter_path: save scalar output parameters/results as one-row CSV.
    time_temperature_path: save returned time-temperature/fitted-curve data CSV.
    Either path may be None.
    """
    scalar_result = {k: v for k, v in result.items() if k != "time_temperature"}

    if parameter_path is not None:
        pd.DataFrame([scalar_result]).to_csv(
            parameter_path, index=False, encoding="utf-8-sig"
        )

    if time_temperature_path is not None:
        data = result.get("time_temperature")
        if data is None:
            raise ValueError(
                "No time_temperature data in result; call fit_tps_kernel with "
                "return_time_temperature=True"
            )
        data.to_csv(time_temperature_path, index=False, encoding="utf-8-sig")

    return scalar_result


# ============================================================
# Thickness correction
# ============================================================

def calculate_thickness_correction(
    lambda_hat: float,
    h_m: float,
    correction_a: float,
    correction_b_um: float,
    correction_c: float,
    min_corrected_lambda: float,
) -> tuple[float, float, float]:
    """
    Calculate thickness-related fitting error and corrected lambda.

    Model:
        error_h = A / (h_um + B_um)^2 + C
        lambda_corrected = lambda_hat - error_h

    Returns:
        h_um
        thickness_error
        lambda_corrected
    """

    h_um = 1.0e6 * h_m

    denominator = (
        h_um + correction_b_um
    ) ** 2

    if denominator <= 0:
        raise ValueError(
            "Invalid thickness-correction denominator"
        )

    thickness_error = (
        correction_a
        / denominator
        + correction_c
    )

    lambda_corrected_raw = (
        lambda_hat
        - thickness_error
    )

    lambda_corrected = max(
        lambda_corrected_raw,
        min_corrected_lambda,
    )

    return (
        h_um,
        thickness_error,
        lambda_corrected,
    )


# ============================================================
# Read one Time-Temperature pair from workbook
# ============================================================

def extract_pair(
    workbook: Path,
    sheet_name: str,
    pair_index: int,
    time_end: float,
) -> tuple[pd.DataFrame, int, int]:

    if not workbook.is_file():
        raise FileNotFoundError(
            f"Data workbook not found: {workbook.name}"
        )

    raw = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        header=None,
    )

    time_column = (
        1
        + 2 * (pair_index - 1)
    )

    temperature_column = (
        time_column + 1
    )

    if temperature_column >= raw.shape[1]:
        raise IndexError(
            f"{workbook.name} does not contain pair "
            f"{pair_index} "
            f"(required Excel columns "
            f"{time_column + 1}-"
            f"{temperature_column + 1})"
        )

    pair = raw.iloc[
        :,
        [
            time_column,
            temperature_column,
        ],
    ].copy()

    pair.columns = [
        "Time",
        "Temperature",
    ]

    pair["Time"] = pd.to_numeric(
        pair["Time"],
        errors="coerce",
    )

    pair["Temperature"] = pd.to_numeric(
        pair["Temperature"],
        errors="coerce",
    )

    pair = pair.dropna(
        subset=[
            "Time",
            "Temperature",
        ]
    )

    pair = pair[
        np.isfinite(pair["Time"])
        & np.isfinite(
            pair["Temperature"]
        )
        & (pair["Time"] >= 0)
        & (pair["Time"] <= time_end)
    ]

    pair = (
        pair
        .sort_values("Time")
        .drop_duplicates("Time")
        .reset_index(drop=True)
    )

    return (
        pair,
        time_column + 1,
        temperature_column + 1,
    )


def resolve_relative_to_script(
    path_text: str | Path,
) -> Path:

    path = Path(path_text)

    return (
        path
        if path.is_absolute()
        else SCRIPT_DIR / path
    )


# ============================================================
# Batch calculation
# ============================================================

def run_batch(
    args: argparse.Namespace,
) -> tuple[int, int]:

    started = time.perf_counter()

    parameter_path = (
        resolve_relative_to_script(
            args.parameter_table
        )
    )

    output_csv = (
        resolve_relative_to_script(
            args.output_csv
        )
    )

    fit_curve_dir = (
        resolve_relative_to_script(
            args.fit_curve_dir
        )
    )

    if not parameter_path.is_file():
        raise FileNotFoundError(
            f"Parameter workbook not found: "
            f"{parameter_path.name}\n"
            f"Put it in the same folder as this script: "
            f"{SCRIPT_DIR}"
        )

    table = pd.read_excel(
        parameter_path
    )

    table.columns = [
        str(column)
        .strip()
        .lower()
        for column in table.columns
    ]

    required = [
        "cv",
        "h",
        "p0",
        "t_end",
        "r",
        "filename",
    ]

    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            "Parameter table is missing columns: "
            + ", ".join(missing)
        )

    if output_csv.exists():
        if args.overwrite:
            output_csv.unlink()
        else:
            raise FileExistsError(
                f"Output already exists: "
                f"{output_csv.name}. "
                "Use --overwrite to replace it."
            )

    if args.save_curves:
        fit_curve_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    pair_counters: defaultdict[
        str,
        int,
    ] = defaultdict(int)

    last_filename: Optional[str] = None
    completed = 0
    failures: list[str] = []

    for row_index, row in table.iterrows():
        excel_row = row_index + 2

        try:
            filename_cell = row[
                "filename"
            ]

            if (
                pd.isna(filename_cell)
                or not str(
                    filename_cell
                ).strip()
            ):
                if last_filename is None:
                    raise ValueError(
                        "filename is blank and no previous "
                        "filename is available"
                    )

                filename = last_filename

            else:
                filename = str(
                    filename_cell
                ).strip()

                if not filename.lower().endswith(
                    ".xlsx"
                ):
                    filename += ".xlsx"

                last_filename = filename

            workbook = (
                SCRIPT_DIR / filename
            )

            key = workbook.name.casefold()

            pair_counters[key] += 1
            pair_index = pair_counters[key]

            t_end = (
                float(row["t_end"])
                if pd.notna(row["t_end"])
                else 5.0
            )

            (
                pair,
                time_col,
                temp_col,
            ) = extract_pair(
                workbook,
                args.sheet_name,
                pair_index,
                t_end,
            )

            opts = Options(
                P0=norm_p0(
                    row["p0"]
                ),
                Cv=norm_cv(
                    row["cv"]
                ),
                r=norm_r(
                    row["r"]
                ),
                h=norm_h(
                    row["h"]
                ),
                t_end=t_end,
                reference_index=(
                    args.reference_index
                ),
                lambda_min=(
                    args.lambda_min
                ),
                lambda_max=(
                    args.lambda_max
                ),
                max_points=(
                    args.max_points
                ),
                sigma_points=(
                    args.sigma_points
                ),
                cpu_dtype=(
                    args.cpu_dtype
                ),
            )

            LOGGER.info(
                "Fitting Excel row %d: %s, pair %d",
                excel_row,
                workbook.name,
                pair_index,
            )

            job_started = (
                time.perf_counter()
            )

            (
                lambda_hat,
                report,
            ) = predict_lambda_grid_cpu(
                pair,
                opts,
            )

            elapsed = (
                time.perf_counter()
                - job_started
            )

            (
                h_um,
                thickness_error,
                lambda_corrected,
            ) = calculate_thickness_correction(
                lambda_hat=lambda_hat,
                h_m=opts.h,
                correction_a=args.correction_a,
                correction_b_um=(
                    args.correction_b_um
                ),
                correction_c=(
                    args.correction_c
                ),
                min_corrected_lambda=(
                    args.min_corrected_lambda
                ),
            )

            if args.save_curves:
                curve_path = (
                    fit_curve_dir
                    / (
                        f"{workbook.stem}"
                        f"_pair{pair_index}"
                        f"_Fit.csv"
                    )
                )

                pd.DataFrame(
                    {
                        "t_s": report["t"],
                        "Tc_calculated_K": (
                            report["Tc"]
                        ),
                        "Te_experimental_K": (
                            report["Te"]
                        ),
                    }
                ).to_csv(
                    curve_path,
                    index=False,
                    encoding="utf-8-sig",
                )

            # Correct unit names:
            # r_mm       : millimetres
            # P0_mW      : milliwatts
            # Cv_J_cm3_K : J cm-3 K-1
            # h_um       : micrometres
            result = {
                "parameter_excel_row": (
                    excel_row
                ),
                "filename": (
                    workbook.name
                ),
                "pair_index": (
                    pair_index
                ),
                "pair_time_column": (
                    time_col
                ),
                "pair_temperature_column": (
                    temp_col
                ),

                "r_mm": (
                    1.0e3 * opts.r
                ),
                "P0_mW": (
                    1.0e3 * opts.P0
                ),
                "t_end_s": (
                    opts.t_end
                ),
                "Cv_J_cm3_K": (
                    1.0e-6 * opts.Cv
                ),
                "h_um": (
                    h_um
                ),

                "lambda_hat_W_mK": (
                    lambda_hat
                ),

                "correction_A": (
                    args.correction_a
                ),
                "correction_b_um": (
                    args.correction_b_um
                ),
                "correction_C_W_mK": (
                    args.correction_c
                ),

                "thickness_error_W_mK": (
                    thickness_error
                ),

                "lambda_corrected_W_mK": (
                    lambda_corrected
                ),

                "final_temperature_K": float(
                    pair[
                        "Temperature"
                    ].iloc[-1]
                ),

                "relative_fit_error": (
                    report["s3_min"]
                ),
            }

            pd.DataFrame(
                [result]
            ).to_csv(
                output_csv,
                mode="a",
                index=False,
                header=not output_csv.exists(),
                encoding=(
                    "utf-8-sig"
                    if not output_csv.exists()
                    else "utf-8"
                ),
            )

            completed += 1

            LOGGER.info(
                "Saved: "
                "lambda_hat=%.6g, "
                "thickness_error=%.6g, "
                "lambda_corrected=%.6g, "
                "Srel=%.6g, "
                "elapsed=%.3f s",
                lambda_hat,
                thickness_error,
                lambda_corrected,
                report["s3_min"],
                elapsed,
            )

        except Exception as exc:
            message = (
                f"Excel row {excel_row} "
                f"failed: {exc}"
            )

            failures.append(
                message
            )

            LOGGER.exception(
                message
            )

    LOGGER.info(
        "Finished: completed=%d, failed=%d, total=%.3f s",
        completed,
        len(failures),
        time.perf_counter() - started,
    )

    if failures:
        LOGGER.error(
            "Failure summary:\n%s",
            "\n".join(failures),
        )

    return (
        completed,
        len(failures),
    )


# ============================================================
# Command-line arguments
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "CPU TPS fitting with thickness correction "
            "using files in the script folder"
        )
    )

    parser.add_argument(
        "--parameter-table",
        default=(
            "Calculation Parameters.xlsx"
        ),
        help=(
            "Parameter workbook name in the same "
            "folder as the script"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "All_fitting_data_CPU_corrected.csv"
        ),
        help=(
            "Output CSV name in the same folder"
        ),
    )

    parser.add_argument(
        "--fit-curve-dir",
        default="Fit_curves_CPU",
        help=(
            "Relative folder name for fitted curves"
        ),
    )

    parser.add_argument(
        "--sheet-name",
        default="T-t",
    )

    parser.add_argument(
        "--lambda-min",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--lambda-max",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--reference-index",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--sigma-points",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--cpu-dtype",
        choices=(
            "float32",
            "float64",
        ),
        default="float64",
        dest="cpu_dtype",
        help=(
            "CPU floating-point precision"
        ),
    )

    # Thickness-correction parameters
    parser.add_argument(
        "--correction-a",
        type=float,
        default=(
            CORRECTION_A_DEFAULT
        ),
        help=(
            "A in error_h = "
            "A/(h_um+B_um)^2 + C"
        ),
    )

    parser.add_argument(
        "--correction-b-um",
        type=float,
        default=(
            CORRECTION_B_UM_DEFAULT
        ),
        help=(
            "B in micrometres for thickness correction"
        ),
    )

    parser.add_argument(
        "--correction-c",
        type=float,
        default=(
            CORRECTION_C_DEFAULT
        ),
        help=(
            "C in W m-1 K-1 for thickness correction"
        ),
    )

    parser.add_argument(
        "--min-corrected-lambda",
        type=float,
        default=(
            MIN_CORRECTED_LAMBDA_DEFAULT
        ),
        help=(
            "Lower physical limit applied to corrected lambda"
        ),
    )

    parser.add_argument(
        "--save-curves",
        action="store_true",
        help=(
            "Save one fitted curve CSV per experiment"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing output CSV"
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = (
        build_parser()
        .parse_args()
    )

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    if (
        not np.isfinite(
            args.correction_a
        )
        or not np.isfinite(
            args.correction_b_um
        )
        or not np.isfinite(
            args.correction_c
        )
    ):
        raise ValueError(
            "Thickness-correction parameters "
            "must be finite"
        )

    if args.correction_b_um <= 0:
        raise ValueError(
            "correction_b_um must be positive"
        )

    if args.min_corrected_lambda <= 0:
        raise ValueError(
            "min_corrected_lambda must be positive"
        )

    check_cpu()

    LOGGER.info(
        "Working folder: %s",
        SCRIPT_DIR,
    )

    LOGGER.info(
        "Thickness correction: "
        "error_h = %.6g/(h_um + %.6g)^2 + %.6g",
        args.correction_a,
        args.correction_b_um,
        args.correction_c,
    )

    run_batch(
        args
    )


if __name__ == "__main__":
    main()
